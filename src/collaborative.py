"""
collaborative.py
----------------
User-based and Item-based Collaborative Filtering
using the Amazon dataset (amazon.csv).

Assumes preprocessing.py has already cleaned the data and saved:
    data/cleaned_amazon.csv   (columns: user_id, product_id, rating)

If that file does not exist yet, this module falls back to reading
the raw amazon.csv and applying minimal cleaning inline.
"""

import os
import ast
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────

def load_data(cleaned_path: str = "data/cleaned_amazon.csv",
              raw_path: str    = "data/amazon.csv") -> pd.DataFrame:
    """
    Load the interaction dataframe.
    Expected columns after loading: user_id, product_id, rating.
    """
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        print(f"[collaborative] Loaded cleaned data: {df.shape}")
    else:
        print(f"[collaborative] cleaned file not found – reading raw CSV …")
        df = _load_raw(raw_path)

    # Keep only the three columns we need
    df = df[["user_id", "product_id", "rating"]].dropna()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(float)
    print(f"[collaborative] Interactions after cleaning: {len(df):,}")
    return df


def _load_raw(path: str) -> pd.DataFrame:
    """Minimal parsing of the raw amazon.csv (multi-user rows)."""
    raw = pd.read_csv(path)

    records = []
    for _, row in raw.iterrows():
        try:
            # user_id and user_name columns hold comma-separated lists
            user_ids = [u.strip() for u in str(row["user_id"]).split(",")]
            rating   = float(str(row["rating"]).replace(",", "").strip())
            pid      = str(row["product_id"]).strip()
            for uid in user_ids:
                if uid:
                    records.append({"user_id": uid,
                                    "product_id": pid,
                                    "rating": rating})
        except Exception:
            continue

    df = pd.DataFrame(records)
    # De-duplicate: one rating per (user, product) – keep the mean
    df = (df.groupby(["user_id", "product_id"], as_index=False)
            .rating.mean())
    return df


# ──────────────────────────────────────────────
# 2.  USER-ITEM MATRIX
# ──────────────────────────────────────────────

def build_user_item_matrix(df: pd.DataFrame):
    """
    Returns
    -------
    matrix   : pd.DataFrame  – rows=users, cols=products, values=rating (NaN = unrated)
    user_idx : dict          – user_id  → row-index
    item_idx : dict          – product_id → col-index
    """
    matrix   = df.pivot_table(index="user_id",
                               columns="product_id",
                               values="rating")
    user_idx = {u: i for i, u in enumerate(matrix.index)}
    item_idx = {p: j for j, p in enumerate(matrix.columns)}
    print(f"[collaborative] Matrix shape: {matrix.shape}  "
          f"(sparsity {100*(1 - df.shape[0]/(matrix.shape[0]*matrix.shape[1])):.1f}%)")
    return matrix, user_idx, item_idx


# ──────────────────────────────────────────────
# 3.  USER-BASED COLLABORATIVE FILTERING
# ──────────────────────────────────────────────

class UserBasedCF:
    """
    Recommends products to a user based on what similar users liked.

    Parameters
    ----------
    k : int   – number of nearest neighbours to consider
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.matrix      = None   # filled user-item matrix (NaN→0 for similarity)
        self.user_index  = None
        self.item_index  = None
        self.sim_matrix  = None   # (n_users × n_users) cosine similarity

    def fit(self, matrix: pd.DataFrame):
        self.matrix     = matrix
        self.user_index = list(matrix.index)
        self.item_index = list(matrix.columns)

        # Fill NaN with 0 for cosine computation
        filled = matrix.fillna(0).values
        self.sim_matrix = cosine_similarity(filled)
        print(f"[UserBasedCF] Fitted on {len(self.user_index)} users.")
        return self

    def recommend(self, user_id: str, n: int = 10,
                  exclude_rated: bool = True) -> pd.DataFrame:
        """
        Returns top-n product recommendations for user_id.
        """
        if user_id not in self.user_index:
            print(f"[UserBasedCF] Unknown user '{user_id}'. Returning popular items.")
            return self._popular_fallback(n)

        u_pos        = self.user_index.index(user_id)
        sim_scores   = self.sim_matrix[u_pos]         # similarity to every user

        # Top-k neighbours (exclude self)
        neighbour_idx = np.argsort(sim_scores)[::-1][1:self.k + 1]
        neighbour_sim = sim_scores[neighbour_idx]

        user_ratings  = self.matrix.iloc[u_pos]       # this user's ratings

        scores = {}
        for col_j, product in enumerate(self.item_index):
            # Skip already-rated items if requested
            if exclude_rated and not np.isnan(user_ratings.iloc[col_j]):
                continue

            # Weighted average of neighbour ratings
            numerator   = 0.0
            denominator = 0.0
            for rank, nb_i in enumerate(neighbour_idx):
                nb_rating = self.matrix.iloc[nb_i, col_j]
                if not np.isnan(nb_rating):
                    numerator   += neighbour_sim[rank] * nb_rating
                    denominator += abs(neighbour_sim[rank])

            if denominator > 0:
                scores[product] = numerator / denominator

        if not scores:
            return self._popular_fallback(n)

        recs = (pd.Series(scores)
                  .sort_values(ascending=False)
                  .head(n)
                  .reset_index())
        recs.columns = ["product_id", "predicted_rating"]
        return recs

    def _popular_fallback(self, n: int) -> pd.DataFrame:
        """Return the n most-rated products as a cold-start fallback."""
        counts = self.matrix.notna().sum().sort_values(ascending=False).head(n)
        df = counts.reset_index()
        df.columns = ["product_id", "predicted_rating"]
        return df


# ──────────────────────────────────────────────
# 4.  ITEM-BASED COLLABORATIVE FILTERING
# ──────────────────────────────────────────────

class ItemBasedCF:
    """
    Recommends products similar to those a user has rated highly.

    Parameters
    ----------
    k : int   – number of most-similar items to use per rated item
    """

    def __init__(self, k: int = 20):
        self.k          = k
        self.matrix     = None
        self.user_index = None
        self.item_index = None
        self.sim_matrix = None   # (n_items × n_items)

    def fit(self, matrix: pd.DataFrame):
        self.matrix     = matrix
        self.user_index = list(matrix.index)
        self.item_index = list(matrix.columns)

        # Compute item-item cosine similarity (transpose the matrix)
        filled = matrix.fillna(0).values.T      # shape: n_items × n_users
        self.sim_matrix = cosine_similarity(filled)
        print(f"[ItemBasedCF] Fitted on {len(self.item_index)} items.")
        return self

    def recommend(self, user_id: str, n: int = 10,
                  exclude_rated: bool = True) -> pd.DataFrame:
        if user_id not in self.user_index:
            print(f"[ItemBasedCF] Unknown user '{user_id}'. Returning popular items.")
            return self._popular_fallback(n)

        u_pos        = self.user_index.index(user_id)
        user_ratings = self.matrix.iloc[u_pos]

        # Items the user has rated
        rated_mask   = user_ratings.notna()
        rated_items  = user_ratings[rated_mask]

        scores = {}
        for target_j, product in enumerate(self.item_index):
            if exclude_rated and rated_mask.iloc[target_j]:
                continue

            # Weighted sum of similarities to rated items
            numerator   = 0.0
            denominator = 0.0
            for rated_product, r in rated_items.items():
                rated_j = self.item_index.index(rated_product)
                sim = self.sim_matrix[target_j, rated_j]
                numerator   += sim * r
                denominator += abs(sim)

            if denominator > 0:
                scores[product] = numerator / denominator

        if not scores:
            return self._popular_fallback(n)

        recs = (pd.Series(scores)
                  .sort_values(ascending=False)
                  .head(n)
                  .reset_index())
        recs.columns = ["product_id", "predicted_rating"]
        return recs

    def _popular_fallback(self, n: int) -> pd.DataFrame:
        counts = self.matrix.notna().sum().sort_values(ascending=False).head(n)
        df = counts.reset_index()
        df.columns = ["product_id", "predicted_rating"]
        return df


# ──────────────────────────────────────────────
# 5.  EVALUATION  (RMSE on held-out ratings)
# ──────────────────────────────────────────────

def evaluate(model, test_df: pd.DataFrame) -> float:
    """
    Predict ratings for (user, product) pairs in test_df.
    Returns RMSE.

    Works for both UserBasedCF and ItemBasedCF.
    """
    actuals    = []
    predicted  = []

    for _, row in test_df.iterrows():
        uid, pid, actual = row["user_id"], row["product_id"], row["rating"]

        if uid not in model.user_index or pid not in model.item_index:
            continue

        u_pos    = model.user_index.index(uid)
        p_pos    = model.item_index.index(pid)

        if isinstance(model, UserBasedCF):
            sim_scores    = model.sim_matrix[u_pos]
            neighbour_idx = np.argsort(sim_scores)[::-1][1:model.k + 1]
            neighbour_sim = sim_scores[neighbour_idx]
            num, den = 0.0, 0.0
            for rank, nb_i in enumerate(neighbour_idx):
                nb_r = model.matrix.iloc[nb_i, p_pos]
                if not np.isnan(nb_r):
                    num += neighbour_sim[rank] * nb_r
                    den += abs(neighbour_sim[rank])
            pred = num / den if den > 0 else np.nan

        else:  # ItemBasedCF
            user_ratings = model.matrix.iloc[u_pos]
            rated_mask   = user_ratings.notna()
            rated_items  = user_ratings[rated_mask]
            num, den = 0.0, 0.0
            for rp, r in rated_items.items():
                rp_j = model.item_index.index(rp)
                sim  = model.sim_matrix[p_pos, rp_j]
                num += sim * r
                den += abs(sim)
            pred = num / den if den > 0 else np.nan

        if not np.isnan(pred):
            actuals.append(actual)
            predicted.append(pred)

    if not actuals:
        print("[evaluate] No overlapping (user, product) pairs in test set.")
        return float("nan")

    rmse = np.sqrt(mean_squared_error(actuals, predicted))
    print(f"[evaluate] RMSE = {rmse:.4f}  (n={len(actuals)} pairs)")
    return rmse


# ──────────────────────────────────────────────
# 6.  CONVENIENCE WRAPPER
# ──────────────────────────────────────────────

def get_collaborative_recommendations(user_id: str,
                                       method: str = "item",
                                       n: int = 10,
                                       data_path: str = "data/amazon.csv"
                                       ) -> pd.DataFrame:
    """
    One-call convenience function used by hybrid.py / app/.

    Parameters
    ----------
    user_id   : str   – target user
    method    : str   – "user" or "item"
    n         : int   – number of recommendations
    data_path : str   – path to raw or cleaned CSV

    Returns
    -------
    pd.DataFrame with columns [product_id, predicted_rating]
    """
    df           = load_data(raw_path=data_path)
    matrix, _, _ = build_user_item_matrix(df)

    if method == "user":
        model = UserBasedCF(k=20).fit(matrix)
    else:
        model = ItemBasedCF(k=20).fit(matrix)

    return model.recommend(user_id, n=n)


# ──────────────────────────────────────────────
# 7.  QUICK DEMO  (run this file directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── load ──────────────────────────────────
    df = load_data(raw_path="data/amazon.csv")

    # ── train / test split ────────────────────
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    matrix, _, _      = build_user_item_matrix(train_df)

    # ── user-based ────────────────────────────
    print("\n=== USER-BASED CF ===")
    ub_model = UserBasedCF(k=20).fit(matrix)

    sample_user = train_df["user_id"].value_counts().index[0]
    print(f"\nTop-10 recommendations for user '{sample_user}':")
    print(ub_model.recommend(sample_user, n=10).to_string(index=False))

    ub_rmse = evaluate(ub_model, test_df)

    # ── item-based ────────────────────────────
    print("\n=== ITEM-BASED CF ===")
    ib_model = ItemBasedCF(k=20).fit(matrix)

    print(f"\nTop-10 recommendations for user '{sample_user}':")
    print(ib_model.recommend(sample_user, n=10).to_string(index=False))

    ib_rmse = evaluate(ib_model, test_df)

    print(f"\nSummary → UserCF RMSE: {ub_rmse:.4f} | ItemCF RMSE: {ib_rmse:.4f}")