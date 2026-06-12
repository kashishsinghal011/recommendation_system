"""
svd.py
------
Matrix Factorization using Singular Value Decomposition (SVD)
via the Surprise library, with a pure-NumPy fallback (TruncatedSVD).

Pipeline
--------
1. Load data  (reuses collaborative.load_data)
2. Train SVD model (Surprise SVD  OR  numpy fallback)
3. Predict rating for any (user, product) pair
4. Generate top-N recommendations for a user
5. Evaluate with RMSE on held-out test set
6. Expose get_svd_recommendations() for hybrid.py / app/
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Reuse the data-loading logic from collaborative.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from collaborative import load_data, build_user_item_matrix


# ──────────────────────────────────────────────
# 1.  SURPRISE SVD  (preferred)
# ──────────────────────────────────────────────

def _train_surprise_svd(df: pd.DataFrame, n_factors: int = 50,
                        n_epochs: int = 20, lr: float = 0.005,
                        reg: float = 0.02):
    """
    Train Surprise SVD and return (algo, trainset).
    Raises ImportError if surprise is not installed.
    """
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split as s_split

    reader  = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data    = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader)
    trainset, _ = s_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
               lr_all=lr, reg_all=reg, random_state=42)
    algo.fit(trainset)
    print(f"[SurpriseSVD] Trained  factors={n_factors}  epochs={n_epochs}")
    return algo, trainset


# ──────────────────────────────────────────────
# 2.  NUMPY FALLBACK SVD
# ──────────────────────────────────────────────

class NumpySVD:
    """
    Lightweight SVD recommender using numpy.linalg.svd.
    Fills missing ratings with user mean before decomposition.

    Parameters
    ----------
    n_components : int  – number of latent factors to keep
    """

    def __init__(self, n_components: int = 50):
        self.k           = n_components
        self.matrix      = None    # original (NaN) matrix
        self.user_index  = None
        self.item_index  = None
        self.predicted   = None    # fully-reconstructed dense matrix

    # ── fit ───────────────────────────────────
    def fit(self, matrix: pd.DataFrame):
        self.matrix     = matrix
        self.user_index = list(matrix.index)
        self.item_index = list(matrix.columns)

        # Fill NaN with per-user mean rating
        filled = matrix.copy()
        user_means = matrix.mean(axis=1)
        for user in matrix.index:
            mask = filled.loc[user].isna()
            filled.loc[user, mask] = user_means[user]
        filled = filled.fillna(filled.mean().mean())   # global mean for all-NaN users

        R = filled.values.astype(float)

        # Truncated SVD
        U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
        k            = min(self.k, len(sigma))
        U_k          = U[:, :k]
        S_k          = np.diag(sigma[:k])
        Vt_k         = Vt[:k, :]

        self.predicted = U_k @ S_k @ Vt_k
        print(f"[NumpySVD] Fitted  components={k}  "
              f"matrix={matrix.shape}")
        return self

    # ── predict one rating ────────────────────
    def predict(self, user_id: str, product_id: str) -> float:
        if user_id not in self.user_index or product_id not in self.item_index:
            return float("nan")
        u = self.user_index.index(user_id)
        p = self.item_index.index(product_id)
        return float(self.predicted[u, p])

    # ── top-N recommendations ─────────────────
    def recommend(self, user_id: str, n: int = 10,
                  exclude_rated: bool = True) -> pd.DataFrame:
        if user_id not in self.user_index:
            print(f"[NumpySVD] Unknown user '{user_id}'. Returning popular items.")
            return self._popular_fallback(n)

        u        = self.user_index.index(user_id)
        pred_row = self.predicted[u]                   # predicted score for every item

        if exclude_rated:
            rated_mask = self.matrix.iloc[u].notna().values
            pred_row   = np.where(rated_mask, -np.inf, pred_row)

        top_idx = np.argsort(pred_row)[::-1][:n]
        recs    = pd.DataFrame({
            "product_id"       : [self.item_index[i] for i in top_idx],
            "predicted_rating" : [pred_row[i]        for i in top_idx]
        })
        return recs

    def _popular_fallback(self, n: int) -> pd.DataFrame:
        counts = self.matrix.notna().sum().sort_values(ascending=False).head(n)
        df = counts.reset_index()
        df.columns = ["product_id", "predicted_rating"]
        return df


# ──────────────────────────────────────────────
# 3.  UNIFIED SVDRecommender CLASS
# ──────────────────────────────────────────────

class SVDRecommender:
    """
    Unified interface — tries Surprise SVD first, falls back to NumpySVD.

    Parameters
    ----------
    n_factors    : int  – latent factors
    n_epochs     : int  – Surprise training epochs (ignored for numpy)
    use_surprise : bool – force Surprise (True) or numpy (False)
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 use_surprise: bool = True):
        self.n_factors    = n_factors
        self.n_epochs     = n_epochs
        self.use_surprise = use_surprise

        self._algo        = None   # Surprise algo
        self._numpy_model = None   # NumpySVD
        self._matrix      = None   # pd.DataFrame (for fallback / evaluation)
        self._df          = None   # raw interaction df
        self._backend     = None   # "surprise" | "numpy"

    # ── fit ───────────────────────────────────
    def fit(self, df: pd.DataFrame):
        self._df     = df
        matrix, _, _ = build_user_item_matrix(df)
        self._matrix = matrix

        if self.use_surprise:
            try:
                self._algo, _ = _train_surprise_svd(
                    df, n_factors=self.n_factors, n_epochs=self.n_epochs)
                self._backend = "surprise"
                return self
            except ImportError:
                print("[SVDRecommender] Surprise not installed – using NumpySVD fallback.")

        self._numpy_model = NumpySVD(n_components=self.n_factors).fit(matrix)
        self._backend     = "numpy"
        return self

    # ── predict one rating ────────────────────
    def predict(self, user_id: str, product_id: str) -> float:
        if self._backend == "surprise":
            pred = self._algo.predict(user_id, product_id)
            return pred.est
        return self._numpy_model.predict(user_id, product_id)

    # ── top-N recommendations ─────────────────
    def recommend(self, user_id: str, n: int = 10,
                  exclude_rated: bool = True) -> pd.DataFrame:
        if self._backend == "surprise":
            return self._surprise_recommend(user_id, n, exclude_rated)
        return self._numpy_model.recommend(user_id, n, exclude_rated)

    def _surprise_recommend(self, user_id: str, n: int,
                             exclude_rated: bool) -> pd.DataFrame:
        all_products = self._matrix.columns.tolist()

        if exclude_rated and user_id in self._matrix.index:
            rated = set(self._matrix.loc[user_id].dropna().index)
        else:
            rated = set()

        scores = []
        for pid in all_products:
            if pid in rated:
                continue
            est = self._algo.predict(user_id, pid).est
            scores.append((pid, est))

        scores.sort(key=lambda x: x[1], reverse=True)
        recs = pd.DataFrame(scores[:n],
                            columns=["product_id", "predicted_rating"])
        return recs

    # ── evaluate (RMSE) ───────────────────────
    def evaluate(self, test_df: pd.DataFrame) -> float:
        actuals, predicted = [], []
        for _, row in test_df.iterrows():
            pred = self.predict(row["user_id"], row["product_id"])
            if not np.isnan(pred):
                actuals.append(row["rating"])
                predicted.append(pred)

        if not actuals:
            print("[SVDRecommender] No overlapping pairs in test set.")
            return float("nan")

        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        print(f"[SVDRecommender] RMSE = {rmse:.4f}  "
              f"(backend={self._backend}, n={len(actuals)})")
        return rmse


# ──────────────────────────────────────────────
# 4.  CONVENIENCE WRAPPER  (for hybrid.py / app/)
# ──────────────────────────────────────────────

def get_svd_recommendations(user_id: str,
                             n: int = 10,
                             n_factors: int = 50,
                             data_path: str = "data/amazon.csv",
                             use_surprise: bool = True) -> pd.DataFrame:
    """
    One-call function for hybrid.py or app/.

    Returns pd.DataFrame with columns [product_id, predicted_rating].
    """
    df    = load_data(raw_path=data_path)
    model = SVDRecommender(n_factors=n_factors,
                           use_surprise=use_surprise).fit(df)
    return model.recommend(user_id, n=n)


# ──────────────────────────────────────────────
# 5.  QUICK DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(raw_path="data/amazon.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("\n=== SVD RECOMMENDER ===")
    model = SVDRecommender(n_factors=50, n_epochs=20,
                           use_surprise=True).fit(train_df)

    sample_user = train_df["user_id"].value_counts().index[0]
    print(f"\nTop-10 recommendations for user '{sample_user}':")
    print(model.recommend(sample_user, n=10).to_string(index=False))

    model.evaluate(test_df)