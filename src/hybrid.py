"""
hybrid.py - Hybrid Recommender System
======================================
Combines three recommendation signals into one final ranking:

  1. SVD   (collaborative filtering via matrix factorization)
  2. NCF   (neural collaborative filtering – NeuMF)
  3. Content-Based (TF-IDF on product name + category + about_product)

Weighting strategy
──────────────────
  final_score = w_svd * svd_score
              + w_ncf * ncf_score
              + w_cb  * cb_score

Weights are tunable; defaults are SVD=0.35, NCF=0.40, CB=0.25.

Dataset : Amazon Product Reviews  (data/amazon.csv)
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ─── Paths & Config ───────────────────────────────────────────────────────────

DATA_PATH      = "data/amazon.csv"
SVD_MODEL_PATH = "models/svd_model.npz"   # saved by svd.py  (optional)
NCF_MODEL_PATH = "models/ncf_model.pt"    # saved by ncf.py  (optional)
MODEL_DIR      = "models"

SEED    = 42
TOP_N   = 10          # default recommendations to return

# Ensemble weights  (must sum to 1.0)
W_SVD = 0.35
W_NCF = 0.40
W_CB  = 0.25

# NCF hyper-params (must match what ncf.py used)
NCF_EMBED_DIM  = 32
NCF_MLP_LAYERS = [64, 32, 16]
NCF_DROPOUT    = 0.2

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_data(path: str):
    """
    Load and clean the Amazon CSV.
    Returns:
        interactions  – long-form DataFrame (user_id, product_id, rating, user_idx, product_idx)
        products      – one row per product with metadata (product_id, product_name, category, about_product)
        user_enc      – fitted LabelEncoder for users
        item_enc      – fitted LabelEncoder for products
        n_users, n_items
    """
    df = pd.read_csv(
        path,
        usecols=["user_id", "product_id", "product_name",
                 "category", "about_product", "rating"]
    )

    # Explode multi-user rows (comma-separated user_ids)
    df["user_id"] = df["user_id"].astype(str).str.split(",")
    df = df.explode("user_id")
    df["user_id"] = df["user_id"].str.strip()

    # Clean rating
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating", "user_id", "product_id"], inplace=True)
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    # Encode IDs
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user_idx"]    = user_enc.fit_transform(df["user_id"])
    df["product_idx"] = item_enc.fit_transform(df["product_id"])

    n_users = df["user_idx"].nunique()
    n_items = df["product_idx"].nunique()

    # Product metadata (deduplicated)
    products = (
        df[["product_id", "product_name", "category", "about_product"]]
        .drop_duplicates("product_id")
        .reset_index(drop=True)
    )
    products["product_idx"] = item_enc.transform(products["product_id"])

    print(f"[Hybrid] Users: {n_users} | Items: {n_items} | Interactions: {len(df)}")
    return df, products, user_enc, item_enc, n_users, n_items


# ─── 2. SVD Scorer ────────────────────────────────────────────────────────────

class SVDScorer:
    """
    Truncated SVD on the user-item rating matrix.
    Produces a full predicted-rating matrix; scores are normalised to [0, 1].
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.pred_matrix: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, n_users: int, n_items: int):
        print("[SVD]  Building user-item matrix …")
        rating_matrix = csr_matrix(
            (df["rating"].values, (df["user_idx"].values, df["product_idx"].values)),
            shape=(n_users, n_items)
        )
        dense   = rating_matrix.toarray().astype(float)
        # Mean-centre by user
        user_means = np.true_divide(
            dense.sum(axis=1),
            (dense != 0).sum(axis=1),
            where=(dense != 0).sum(axis=1) != 0
        )
        user_means = np.nan_to_num(user_means)
        dense_centred = dense.copy()
        for u in range(n_users):
            mask = dense[u] != 0
            dense_centred[u][mask] -= user_means[u]

        k = min(self.n_factors, min(dense_centred.shape) - 1)
        U, sigma, Vt = svds(dense_centred, k=k)
        pred = np.dot(np.dot(U, np.diag(sigma)), Vt)
        pred += user_means.reshape(-1, 1)

        # Normalise to [0, 1]
        scaler = MinMaxScaler()
        self.pred_matrix = scaler.fit_transform(pred)
        print("[SVD]  Done.")
        return self

    def score(self, user_idx: int) -> np.ndarray:
        """Returns normalised scores for all items for a given user index."""
        if self.pred_matrix is None:
            raise RuntimeError("SVDScorer not fitted yet.")
        return self.pred_matrix[user_idx]


# ─── 3. NCF Scorer ────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """Mirrors the NeuMF architecture in ncf.py exactly."""

    def __init__(self, n_users, n_items, embed_dim, mlp_layers, dropout):
        super().__init__()
        self.gmf_user_emb = nn.Embedding(n_users, embed_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embed_dim)
        self.mlp_user_emb = nn.Embedding(n_users, embed_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embed_dim)

        layers, in_dim = [], embed_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp    = nn.Sequential(*layers)
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        gmf_out = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)
        mlp_out = self.mlp(torch.cat([self.mlp_user_emb(user_ids),
                                      self.mlp_item_emb(item_ids)], dim=-1))
        return self.sigmoid(self.output(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1))


class NCFScorer:
    """Wraps the NeuMF model and returns normalised per-item scores."""

    def __init__(self, n_users: int, n_items: int):
        self.n_items = n_items
        self.model   = NeuMF(n_users, n_items,
                             NCF_EMBED_DIM, NCF_MLP_LAYERS, NCF_DROPOUT).to(DEVICE)
        self._scaler = MinMaxScaler()

    def load(self, path: str):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"[NCF]  Loaded weights from {path}")
        else:
            print(f"[NCF]  No saved model at {path} — using random weights (train first).")
        self.model.eval()
        return self

    def score(self, user_idx: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([user_idx] * self.n_items, dtype=torch.long).to(DEVICE)
            i = torch.arange(self.n_items, dtype=torch.long).to(DEVICE)
            scores = self.model(u, i).cpu().numpy()
        # Normalise to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores


# ─── 4. Content-Based Scorer ──────────────────────────────────────────────────

class ContentScorer:
    """
    TF-IDF on (product_name + category + about_product).
    Given a user's interaction history, builds a profile vector and
    returns cosine similarity against all items.
    """

    def __init__(self):
        self.tfidf      = TfidfVectorizer(max_features=5000, stop_words="english",
                                          ngram_range=(1, 2))
        self.tfidf_matrix: np.ndarray | None = None
        self.product_idx_map: dict            = {}   # product_idx → row in tfidf_matrix

    def fit(self, products: pd.DataFrame):
        print("[CB]   Building TF-IDF matrix …")
        products = products.copy()
        for col in ["product_name", "category", "about_product"]:
            products[col] = products[col].fillna("").astype(str)

        products["text"] = (
            products["product_name"] + " " +
            products["category"].str.replace("|", " ", regex=False) + " " +
            products["about_product"]
        )
        self.tfidf_matrix = self.tfidf.fit_transform(products["text"]).toarray()
        self.product_idx_map = dict(zip(products["product_idx"], range(len(products))))
        print(f"[CB]   TF-IDF shape: {self.tfidf_matrix.shape}")
        return self

    def score(self, user_history_idxs: list[int], n_items: int) -> np.ndarray:
        """
        user_history_idxs : list of product_idx values the user interacted with.
        Returns normalised cosine-similarity scores for all n_items.
        """
        if not user_history_idxs:
            return np.zeros(n_items)

        rows = [self.product_idx_map[i]
                for i in user_history_idxs if i in self.product_idx_map]
        if not rows:
            return np.zeros(n_items)

        profile     = self.tfidf_matrix[rows].mean(axis=0, keepdims=True)
        sims        = cosine_similarity(profile, self.tfidf_matrix).flatten()

        # Map back to product_idx order
        out = np.zeros(n_items)
        for prod_idx, row in self.product_idx_map.items():
            out[prod_idx] = sims[row]
        return out


# ─── 5. Hybrid Recommender ────────────────────────────────────────────────────

class HybridRecommender:
    """
    Orchestrates SVD, NCF, and Content-Based scorers.
    Exposes .fit() and .recommend() as the public API.
    """

    def __init__(self,
                 w_svd: float = W_SVD,
                 w_ncf: float = W_NCF,
                 w_cb:  float = W_CB):
        assert abs(w_svd + w_ncf + w_cb - 1.0) < 1e-6, "Weights must sum to 1."
        self.w_svd = w_svd
        self.w_ncf = w_ncf
        self.w_cb  = w_cb

        # filled in .fit()
        self.df        = None
        self.products  = None
        self.user_enc  = None
        self.item_enc  = None
        self.n_users   = None
        self.n_items   = None
        self.svd       = None
        self.ncf       = None
        self.cb        = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, data_path: str = DATA_PATH,
            ncf_weights: str = NCF_MODEL_PATH) -> "HybridRecommender":

        # Load data
        (self.df, self.products,
         self.user_enc, self.item_enc,
         self.n_users, self.n_items) = load_data(data_path)

        # SVD
        self.svd = SVDScorer(n_factors=50).fit(self.df, self.n_users, self.n_items)

        # NCF
        self.ncf = NCFScorer(self.n_users, self.n_items).load(ncf_weights)

        # Content-Based
        self.cb = ContentScorer().fit(self.products)

        print("\n[Hybrid] All scorers ready.")
        print(f"         Weights → SVD={self.w_svd} | NCF={self.w_ncf} | CB={self.w_cb}")
        return self

    # ── recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_id: str, top_n: int = TOP_N,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            rank, product_id, product_name, category,
            svd_score, ncf_score, cb_score, hybrid_score
        """
        # Resolve user index
        try:
            user_idx = int(self.user_enc.transform([user_id])[0])
        except ValueError:
            print(f"[Hybrid] Unknown user '{user_id}' — returning popular items.")
            return self._fallback(top_n)

        # Scores from each model (all normalised [0,1])
        svd_scores = self.svd.score(user_idx)
        ncf_scores = self.ncf.score(user_idx)

        # User history for CB
        history = self.df[self.df["user_idx"] == user_idx]["product_idx"].tolist()
        cb_scores  = self.cb.score(history, self.n_items)

        # Weighted ensemble
        hybrid = (self.w_svd * svd_scores +
                  self.w_ncf * ncf_scores +
                  self.w_cb  * cb_scores)

        # Exclude already-seen items
        if exclude_seen:
            hybrid[history] = -1.0

        top_indices = np.argsort(-hybrid)[:top_n]
        product_ids = self.item_enc.inverse_transform(top_indices)

        # Build output DataFrame
        id_to_meta = self.products.set_index("product_id")[["product_name", "category"]]
        rows = []
        for rank, (idx, pid) in enumerate(zip(top_indices, product_ids), 1):
            meta = id_to_meta.loc[pid] if pid in id_to_meta.index else pd.Series({"product_name": "N/A", "category": "N/A"})
            rows.append({
                "rank":         rank,
                "product_id":   pid,
                "product_name": meta["product_name"][:60] + "…" if len(str(meta["product_name"])) > 60 else meta["product_name"],
                "category":     str(meta["category"]).split("|")[0],
                "svd_score":    round(float(svd_scores[idx]), 4),
                "ncf_score":    round(float(ncf_scores[idx]), 4),
                "cb_score":     round(float(cb_scores[idx]),  4),
                "hybrid_score": round(float(hybrid[idx]),     4),
            })

        return pd.DataFrame(rows)

    # ── fallback: popularity-based ─────────────────────────────────────────────

    def _fallback(self, top_n: int) -> pd.DataFrame:
        popular = (
            self.df.groupby("product_id")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "n_ratings"})
            .sort_values(["avg_rating", "n_ratings"], ascending=False)
            .head(top_n)
            .reset_index()
        )
        popular.insert(0, "rank", range(1, len(popular) + 1))
        return popular

    # ── evaluate (RMSE on held-out ratings) ───────────────────────────────────

    def evaluate_rmse(self, sample_n: int = 2000) -> float:
        """
        Samples interactions, generates hybrid scores, maps back to [1,5]
        rating scale, and computes RMSE vs true ratings.
        """
        sample = self.df.sample(min(sample_n, len(self.df)), random_state=SEED)
        preds, actuals = [], []

        for _, row in sample.iterrows():
            u = int(row["user_idx"])
            i = int(row["product_idx"])

            svd_s = float(self.svd.score(u)[i])
            ncf_s = float(self.ncf.score(u)[i])
            history = self.df[self.df["user_idx"] == u]["product_idx"].tolist()
            cb_s  = float(self.cb.score(history, self.n_items)[i])

            hybrid = self.w_svd * svd_s + self.w_ncf * ncf_s + self.w_cb * cb_s
            # Scale [0,1] → [1,5]
            pred_rating = 1 + hybrid * 4
            preds.append(pred_rating)
            actuals.append(float(row["rating"]))

        rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))
        print(f"[Hybrid] RMSE on {len(sample)} samples: {rmse:.4f}")
        return rmse


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("   Hybrid Recommender  (SVD + NCF + Content-Based)")
    print("=" * 65)

    # Build & fit
    recommender = HybridRecommender(w_svd=W_SVD, w_ncf=W_NCF, w_cb=W_CB)
    recommender.fit(DATA_PATH, NCF_MODEL_PATH)

    # ── Demo: recommend for a known user ─────────────────────────────────────
    sample_user = recommender.df["user_id"].iloc[0]
    print(f"\n[Hybrid] Top-{TOP_N} recommendations for user: {sample_user}\n")
    recs = recommender.recommend(sample_user, top_n=TOP_N)
    print(recs.to_string(index=False))

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print()
    recommender.evaluate_rmse(sample_n=1000)

    # ── Compare weights (ablation) ───────────────────────────────────────────
    print("\n[Hybrid] Ablation — weight sensitivity:")
    configs = [
        ("SVD-only",   1.0,  0.0,  0.0),
        ("NCF-only",   0.0,  1.0,  0.0),
        ("CB-only",    0.0,  0.0,  1.0),
        ("Equal",      0.33, 0.34, 0.33),
        ("Default",    W_SVD, W_NCF, W_CB),
    ]
    for label, ws, wn, wc in configs:
        r = HybridRecommender(w_svd=ws, w_ncf=wn, w_cb=wc)
        r.df       = recommender.df
        r.products = recommender.products
        r.user_enc = recommender.user_enc
        r.item_enc = recommender.item_enc
        r.n_users  = recommender.n_users
        r.n_items  = recommender.n_items
        r.svd      = recommender.svd
        r.ncf      = recommender.ncf
        r.cb       = recommender.cb
        print(f"  {label:<12}", end="  ")
        r.evaluate_rmse(sample_n=500)

    print("\n[Hybrid] Done.")


if __name__ == "__main__":
    main()