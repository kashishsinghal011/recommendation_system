"""
popularity.py
-------------
Popularity-based recommender (non-personalised baseline).

Scores each item by a Bayesian average rating that accounts for
both average rating and number of ratings — so a product with
1000 ratings at 4.0 ranks above one with 3 ratings at 5.0.

    score = (v / (v + m)) * R  +  (m / (v + m)) * C

    where:
        R = item's mean rating
        v = item's rating count
        m = minimum rating count threshold (e.g. 50th percentile)
        C = global mean rating across all items

Usage:
    python src/popularity.py                    # top-10 overall
    python src/popularity.py --category "Electronics" --top 20
"""

import os
import argparse
import logging

import pandas as pd
import numpy as np
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD PROCESSED DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path   = os.path.join(PROCESSED_DIR, "train.csv")
    products_path = os.path.join(PROCESSED_DIR, "products.csv")

    log.info(f"Loading train data from  {train_path}")
    train = pd.read_csv(train_path)

    log.info(f"Loading product metadata from  {products_path}")
    products = pd.read_csv(products_path)

    return train, products


# ═════════════════════════════════════════════════════════════════════════════
# 2.  COMPUTE BAYESIAN POPULARITY SCORE
# ═════════════════════════════════════════════════════════════════════════════

def compute_popularity(train: pd.DataFrame,
                       products: pd.DataFrame,
                       percentile: float = 0.50) -> pd.DataFrame:
    """
    Aggregate per-item stats from the interaction data, then join with
    product metadata and compute the Bayesian average score.
    """
    log.info("Computing per-item statistics …")

    # Per-item stats from interactions
    item_stats = (
        train.groupby("product_id")["rating"]
        .agg(mean_rating="mean", vote_count="count")
        .reset_index()
    )

    # Global mean and minimum-vote threshold
    C = item_stats["mean_rating"].mean()
    m = item_stats["vote_count"].quantile(percentile)
    log.info(f"Global mean rating C = {C:.4f}  |  Min-vote threshold m = {m:.0f}")

    # Bayesian average
    v = item_stats["vote_count"]
    R = item_stats["mean_rating"]
    item_stats["popularity_score"] = (v / (v + m)) * R + (m / (v + m)) * C

    # Merge with product metadata (name, category, price, img_link)
    keep_cols = ["product_id", "product_name", "category",
                 "discounted_price", "actual_price",
                 "discount_percentage", "img_link", "product_link"]
    keep_cols = [c for c in keep_cols if c in products.columns]

    pop_df = item_stats.merge(products[keep_cols], on="product_id", how="left")
    pop_df = pop_df.sort_values("popularity_score", ascending=False).reset_index(drop=True)
    pop_df["rank"] = pop_df.index + 1

    log.info(f"Popularity table built: {len(pop_df)} items.")
    return pop_df, C, m


# ═════════════════════════════════════════════════════════════════════════════
# 3.  RECOMMENDER CLASS
# ═════════════════════════════════════════════════════════════════════════════

class PopularityRecommender:
    """
    Recommend the globally (or category-) most popular items.

    Attributes
    ----------
    pop_df : pd.DataFrame
        Full ranked popularity table.
    C, m   : float
        Global mean and vote threshold used to compute scores.
    """

    def __init__(self, pop_df: pd.DataFrame, C: float, m: float):
        self.pop_df = pop_df
        self.C      = C
        self.m      = m

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self,
                  top_n: int = 10,
                  category: str = None) -> pd.DataFrame:
        """
        Return the top-N most popular products.

        Parameters
        ----------
        top_n    : number of recommendations to return
        category : optional top-level category filter (case-insensitive substring)
        """
        df = self.pop_df.copy()

        if category:
            mask = df["category"].str.contains(category, case=False, na=False)
            df   = df[mask].reset_index(drop=True)
            df["rank"] = df.index + 1
            if df.empty:
                log.warning(f"No products found for category='{category}'")
                return df

        result = df.head(top_n)[
            ["rank", "product_id", "product_name", "category",
             "mean_rating", "vote_count", "popularity_score",
             "discounted_price", "actual_price", "discount_percentage"]
        ]
        return result

    # ── Evaluate (hit-rate against leave-one-out test set) ────────────────────

    def evaluate(self, test: pd.DataFrame, top_n: int = 10) -> dict:
        """
        Hit Rate @ N  —  fraction of test users whose held-out item
        appears in the top-N popular items.
        """
        top_items = set(self.pop_df.head(top_n)["product_id"])
        hits = test["product_id"].isin(top_items).sum()
        hr   = hits / len(test)
        log.info(f"Hit Rate @{top_n}: {hr:.4f}  ({hits}/{len(test)} users)")
        return {"hit_rate": hr, "hits": int(hits), "total_users": len(test)}

    # ── Serialise ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PopularityRecommender":
        model = joblib.load(path)
        log.info(f"Model loaded from {path}")
        return model


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(top_n: int = 10, category: str = None):
    # Load
    train, products = load_data()

    # Build popularity table
    pop_df, C, m = compute_popularity(train, products)

    # Instantiate recommender
    model = PopularityRecommender(pop_df, C, m)

    # Evaluate on test set
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        model.evaluate(test, top_n=top_n)

    # Save model
    model_path = os.path.join(MODELS_DIR, "popularity_model.pkl")
    model.save(model_path)

    # Print recommendations
    recs = model.recommend(top_n=top_n, category=category)
    print(f"\n{'='*60}")
    print(f"  Top-{top_n} Popular Products"
          + (f"  [category: {category}]" if category else ""))
    print(f"{'='*60}")
    print(recs.to_string(index=False))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popularity-based recommender")
    parser.add_argument("--top",      type=int,   default=10,  help="Number of recommendations")
    parser.add_argument("--category", type=str,   default=None, help="Filter by category name")
    args = parser.parse_args()

    main(top_n=args.top, category=args.category)