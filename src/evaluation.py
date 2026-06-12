"""
evaluation.py - Unified Evaluation Suite
==========================================
Evaluates all recommender models built in this project:

  1. Popularity Baseline
  2. Content-Based  (TF-IDF cosine similarity)
  3. Collaborative Filtering  (User-User / Item-Item)
  4. SVD  (Matrix Factorization)
  5. NCF  (Neural Collaborative Filtering — NeuMF)
  6. Hybrid  (SVD + NCF + Content-Based)

Metrics computed
────────────────
  Rating-prediction  : RMSE, MAE
  Ranking            : Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR@K
  Coverage           : Catalog Coverage, Intra-list Diversity

Usage
─────
  python src/evaluation.py              # evaluate all models, K=10
  python src/evaluation.py --model svd  # single model
  python src/evaluation.py --k 5        # change K
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

warnings.filterwarnings("ignore")

# Torch is optional — NCF / Hybrid are skipped gracefully if unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─── Paths & Global Config ────────────────────────────────────────────────────

DATA_PATH      = "data/amazon.csv"
NCF_MODEL_PATH = "models/ncf_model.pt"

SEED      = 42
K_DEFAULT = 10          # top-K for ranking metrics
REL_THRESHOLD = 3.5     # rating >= this is "relevant"
EVAL_USERS    = 300     # max users to evaluate (speed vs accuracy)

# NCF arch (must match ncf.py)
NCF_EMBED_DIM  = 32
NCF_MLP_LAYERS = [64, 32, 16]
NCF_DROPOUT    = 0.2

np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(
        path,
        usecols=["user_id", "product_id", "product_name",
                 "category", "about_product", "rating"]
    )
    df["user_id"] = df["user_id"].astype(str).str.split(",")
    df = df.explode("user_id")
    df["user_id"] = df["user_id"].str.strip()
    df["rating"]  = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating", "user_id", "product_id"], inplace=True)
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)].reset_index(drop=True)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user_idx"]    = user_enc.fit_transform(df["user_id"])
    df["product_idx"] = item_enc.fit_transform(df["product_id"])

    n_users = df["user_idx"].nunique()
    n_items = df["product_idx"].nunique()

    products = (
        df[["product_id", "product_name", "category", "about_product", "product_idx"]]
        .drop_duplicates("product_id")
        .reset_index(drop=True)
    )

    print(f"[Data]  Users={n_users} | Items={n_items} | Interactions={len(df)}")
    return df, products, user_enc, item_enc, n_users, n_items


def train_test_split_by_user(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    For each user, hold out the last `test_ratio` fraction of interactions
    (sorted by row order) as the test set.
    """
    train_rows, test_rows = [], []
    for _, group in df.groupby("user_idx"):
        n = len(group)
        split = max(1, int(n * (1 - test_ratio)))
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])
    return pd.concat(train_rows).reset_index(drop=True), \
           pd.concat(test_rows).reset_index(drop=True)


# ─── Metric Functions ─────────────────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top = recommended[:k]
    return len(set(top) & relevant) / k if k else 0.0

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    top = recommended[:k]
    return len(set(top) & relevant) / len(relevant) if relevant else 0.0

def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return 1.0 if set(recommended[:k]) & relevant else 0.0

def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

def mrr_at_k(recommended: list, relevant: set, k: int) -> float:
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def catalog_coverage(all_recs, n_items: int) -> float:
    unique = len(set(item for recs in all_recs for item in recs))
    return unique / n_items if n_items else 0.0

def intra_list_diversity(recs: list, tfidf_matrix: np.ndarray,
                         product_idx_map: dict) -> float:
    """Average pairwise dissimilarity (1 - cosine_sim) within a recommendation list."""
    rows = [product_idx_map[i] for i in recs if i in product_idx_map]
    if len(rows) < 2:
        return 0.0
    vecs = tfidf_matrix[rows]
    sims = cosine_similarity(vecs)
    n    = len(rows)
    avg_sim = (sims.sum() - n) / (n * (n - 1))   # exclude diagonal
    return float(1.0 - avg_sim)


# ─── Score Aggregator ─────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Runs ranking & rating metrics for a score-producing callable.

    score_fn(user_idx) -> np.ndarray of shape (n_items,)  [higher = more relevant]
    """

    def __init__(self, score_fn, train_df, test_df, n_items,
                 tfidf_matrix=None, product_idx_map=None):
        self.score_fn        = score_fn
        self.train_df        = train_df
        self.test_df         = test_df
        self.n_items         = n_items
        self.tfidf_matrix    = tfidf_matrix
        self.product_idx_map = product_idx_map

    def evaluate(self, k: int = K_DEFAULT, max_users: int = EVAL_USERS) -> dict:
        users = self.test_df["user_idx"].unique()
        if len(users) > max_users:
            users = np.random.choice(users, max_users, replace=False)

        prec, rec, hit, ndcg_vals, mrr_vals, div_vals = [], [], [], [], [], []
        all_recs = []

        for user_idx in users:
            # Relevant items in test set
            test_user = self.test_df[self.test_df["user_idx"] == user_idx]
            relevant  = set(
                test_user[test_user["rating"] >= REL_THRESHOLD]["product_idx"].tolist()
            )
            if not relevant:
                continue

            # Seen in train → exclude
            seen = set(
                self.train_df[self.train_df["user_idx"] == user_idx]["product_idx"].tolist()
            )

            scores = self.score_fn(int(user_idx)).copy()
            if seen:
                scores[list(seen)] = -np.inf
            top_k  = list(np.argsort(-scores)[:k])
            all_recs.append(top_k)

            prec.append(precision_at_k(top_k, relevant, k))
            rec.append(recall_at_k(top_k, relevant, k))
            hit.append(hit_rate_at_k(top_k, relevant, k))
            ndcg_vals.append(ndcg_at_k(top_k, relevant, k))
            mrr_vals.append(mrr_at_k(top_k, relevant, k))

            if self.tfidf_matrix is not None and self.product_idx_map is not None:
                div_vals.append(intra_list_diversity(top_k, self.tfidf_matrix,
                                                     self.product_idx_map))

        results = {
            f"Precision@{k}":  float(np.mean(prec))      if prec else 0.0,
            f"Recall@{k}":     float(np.mean(rec))       if rec else 0.0,
            f"HitRate@{k}":    float(np.mean(hit))       if hit else 0.0,
            f"NDCG@{k}":       float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
            f"MRR@{k}":        float(np.mean(mrr_vals))  if mrr_vals else 0.0,
            f"Coverage":       catalog_coverage(all_recs, self.n_items),
        }
        if div_vals:
            results["Diversity"] = float(np.mean(div_vals))
        return results


# ─── Individual Model Score Functions ────────────────────────────────────────

# ── Popularity ────────────────────────────────────────────────────────────────

def build_popularity_scorer(train_df, n_items):
    pop_scores = np.zeros(n_items)
    agg = train_df.groupby("product_idx")["rating"].mean()
    for idx, score in agg.items():
        pop_scores[idx] = score
    scaler = MinMaxScaler()
    pop_scores = scaler.fit_transform(pop_scores.reshape(-1, 1)).flatten()
    return lambda user_idx: pop_scores.copy()   # same for every user


# ── Content-Based ─────────────────────────────────────────────────────────────

def build_cb_scorer(train_df, products, n_items):
    for col in ["product_name", "category", "about_product"]:
        products[col] = products[col].fillna("").astype(str)
    products["text"] = (
        products["product_name"] + " " +
        products["category"].str.replace("|", " ", regex=False) + " " +
        products["about_product"]
    )
    tfidf    = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix   = tfidf.fit_transform(products["text"]).toarray()
    idx_map  = dict(zip(products["product_idx"], range(len(products))))

    def score_fn(user_idx):
        history = train_df[train_df["user_idx"] == user_idx]["product_idx"].tolist()
        if not history:
            return np.zeros(n_items)
        rows    = [idx_map[i] for i in history if i in idx_map]
        if not rows:
            return np.zeros(n_items)
        profile = matrix[rows].mean(axis=0, keepdims=True)
        sims    = cosine_similarity(profile, matrix).flatten()
        out     = np.zeros(n_items)
        for pidx, row in idx_map.items():
            out[pidx] = sims[row]
        return out

    return score_fn, matrix, idx_map


# ── Collaborative Filtering (Item-Item) ───────────────────────────────────────

def build_cf_scorer(train_df, n_users, n_items):
    mat = csr_matrix(
        (train_df["rating"].values,
         (train_df["user_idx"].values, train_df["product_idx"].values)),
        shape=(n_users, n_items)
    ).toarray().astype(float)

    # Item-item cosine similarity
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms[norms == 0] = 1
    normed    = mat / norms
    item_sim  = normed.T @ normed   # (n_items, n_items)

    def score_fn(user_idx):
        user_vec = mat[user_idx]
        scores   = item_sim @ user_vec
        scaler   = MinMaxScaler()
        return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    return score_fn


# ── SVD ───────────────────────────────────────────────────────────────────────

def build_svd_scorer(train_df, n_users, n_items, n_factors=50):
    mat = csr_matrix(
        (train_df["rating"].values,
         (train_df["user_idx"].values, train_df["product_idx"].values)),
        shape=(n_users, n_items)
    ).toarray().astype(float)

    user_means = np.true_divide(
        mat.sum(axis=1),
        (mat != 0).sum(axis=1),
        where=(mat != 0).sum(axis=1) != 0
    )
    user_means = np.nan_to_num(user_means)
    centred    = mat.copy()
    for u in range(n_users):
        mask = mat[u] != 0
        centred[u][mask] -= user_means[u]

    k  = min(n_factors, min(centred.shape) - 1)
    U, sigma, Vt = svds(centred, k=k)
    pred = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred += user_means.reshape(-1, 1)

    scaler = MinMaxScaler()
    pred   = scaler.fit_transform(pred)

    return lambda user_idx: pred[user_idx]


# ── NCF ───────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class NeuMF(nn.Module):
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

        def forward(self, u, i):
            gmf = self.gmf_user_emb(u) * self.gmf_item_emb(i)
            mlp = self.mlp(torch.cat([self.mlp_user_emb(u), self.mlp_item_emb(i)], dim=-1))
            return self.sigmoid(self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1))


def build_ncf_scorer(n_users, n_items, weights_path):
    if not TORCH_AVAILABLE:
        print("[NCF]  torch not available — NCF scorer disabled (returns zeros).")
        return lambda user_idx: np.zeros(n_items)

    model = NeuMF(n_users, n_items, NCF_EMBED_DIM, NCF_MLP_LAYERS, NCF_DROPOUT).to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"[NCF]  Loaded weights from {weights_path}")
    else:
        print(f"[NCF]  No weights found at {weights_path} — using random init.")
    model.eval()

    def score_fn(user_idx):
        with torch.no_grad():
            u = torch.tensor([user_idx] * n_items, dtype=torch.long).to(DEVICE)
            i = torch.arange(n_items, dtype=torch.long).to(DEVICE)
            s = model(u, i).cpu().numpy()
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    return score_fn


# ── Hybrid ────────────────────────────────────────────────────────────────────

def build_hybrid_scorer(svd_fn, ncf_fn, cb_fn,
                        w_svd=0.35, w_ncf=0.40, w_cb=0.25):
    def score_fn(user_idx):
        return (w_svd * svd_fn(user_idx) +
                w_ncf * ncf_fn(user_idx) +
                w_cb  * cb_fn(user_idx))
    return score_fn


# ── RMSE / MAE (rating prediction) ────────────────────────────────────────────

def rating_metrics(score_fn, test_df, sample_n=2000) -> dict:
    sample = test_df.sample(min(sample_n, len(test_df)), random_state=SEED)
    preds, actuals = [], []
    for _, row in sample.iterrows():
        s = float(score_fn(int(row["user_idx"]))[int(row["product_idx"])])
        preds.append(1 + s * 4)        # scale [0,1] → [1,5]
        actuals.append(float(row["rating"]))
    return {
        "RMSE": round(rmse(np.array(actuals), np.array(preds)), 4),
        "MAE":  round(mae(np.array(actuals),  np.array(preds)), 4),
    }


# ─── Pretty Print ─────────────────────────────────────────────────────────────

def print_results(name: str, metrics: dict):
    bar = "─" * 52
    print(f"\n┌{bar}┐")
    print(f"│  Model : {name:<41}│")
    print(f"├{bar}┤")
    for k, v in metrics.items():
        print(f"│  {k:<20}  {v:.4f}{'':>23}│")
    print(f"└{bar}┘")


def print_comparison_table(results):
    all_keys = list(next(iter(results.values())).keys())
    col_w    = 14
    header   = f"{'Model':<18}" + "".join(f"{k:>{col_w}}" for k in all_keys)
    print("\n" + "=" * len(header))
    print("  FULL COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        row = f"{model_name:<18}" + "".join(
            f"{metrics.get(k, 0.0):>{col_w}.4f}" for k in all_keys
        )
        print(row)
    print("=" * len(header))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recommender System Evaluator")
    parser.add_argument("--model", default="all",
                        choices=["all", "popularity", "cb", "cf", "svd", "ncf", "hybrid"],
                        help="Which model to evaluate")
    parser.add_argument("--k", type=int, default=K_DEFAULT, help="Top-K for ranking metrics")
    parser.add_argument("--users", type=int, default=EVAL_USERS, help="Max users to evaluate")
    parser.add_argument("--data", default=DATA_PATH, help="Path to amazon.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("   Recommender System — Evaluation Suite")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    df, products, user_enc, item_enc, n_users, n_items = load_data(args.data)
    train_df, test_df = train_test_split_by_user(df, test_ratio=0.2)
    print(f"[Eval]  Train={len(train_df)} | Test={len(test_df)} | K={args.k}")

    # ── Build scorers ─────────────────────────────────────────────────────────
    print("\n[Eval]  Building models …")
    pop_fn  = build_popularity_scorer(train_df, n_items)
    cb_fn, tfidf_mat, idx_map = build_cb_scorer(train_df, products.copy(), n_items)
    cf_fn   = build_cf_scorer(train_df, n_users, n_items)
    svd_fn  = build_svd_scorer(train_df, n_users, n_items)
    ncf_fn  = build_ncf_scorer(n_users, n_items, NCF_MODEL_PATH)
    hyb_fn  = build_hybrid_scorer(svd_fn, ncf_fn, cb_fn)

    model_map = {
        "popularity": pop_fn,
        "cb":         cb_fn,
        "cf":         cf_fn,
        "svd":        svd_fn,
        "ncf":        ncf_fn,
        "hybrid":     hyb_fn,
    }

    to_run = list(model_map.keys()) if args.model == "all" else [args.model]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results = {}

    for name in to_run:
        print(f"\n[Eval]  Evaluating: {name.upper()} …")
        fn = model_map[name]

        evaluator = ModelEvaluator(
            score_fn        = fn,
            train_df        = train_df,
            test_df         = test_df,
            n_items         = n_items,
            tfidf_matrix    = tfidf_mat,
            product_idx_map = idx_map,
        )

        ranking  = evaluator.evaluate(k=args.k, max_users=args.users)
        rating   = rating_metrics(fn, test_df)
        combined = {**rating, **ranking}
        all_results[name] = combined
        print_results(name.upper(), combined)

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(to_run) > 1:
        print_comparison_table(all_results)

    # ── Best model summary ────────────────────────────────────────────────────
    if len(to_run) > 1:
        best_ndcg = max(all_results, key=lambda m: all_results[m].get(f"NDCG@{args.k}", 0))
        best_hr   = max(all_results, key=lambda m: all_results[m].get(f"HitRate@{args.k}", 0))
        low_rmse  = min(all_results, key=lambda m: all_results[m].get("RMSE", 9999))
        print(f"\n[Summary]")
        print(f"  Best NDCG@{args.k}    → {best_ndcg.upper()}")
        print(f"  Best HitRate@{args.k} → {best_hr.upper()}")
        print(f"  Lowest RMSE       → {low_rmse.upper()}")

    print("\n[Eval]  Evaluation complete.")


if __name__ == "__main__":
    main()