"""
preprocessing.py
----------------
Cleans and prepares the Amazon product dataset for the recommender system.

Pipeline:
  1. Load raw CSV
  2. Clean price / discount / rating columns
  3. Explode multi-valued user_id & review_id into one row per (user, product)
  4. Build a clean user–item interaction matrix (ratings)
  5. Engineer content features (TF-IDF on text, category encoding)
  6. Save processed artefacts to  data/processed/

Usage:
  python src/preprocessing.py                   # default path
  python src/preprocessing.py --data data/amazon.csv
"""

import os
import re
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = os.path.join("data", "amazon.csv")
PROCESSED_DIR   = os.path.join("data", "processed")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_USER_INTERACTIONS  = 2   # drop users with fewer interactions
MIN_ITEM_INTERACTIONS  = 2   # drop items with fewer interactions
TFIDF_MAX_FEATURES     = 5000
RANDOM_STATE           = 42


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ═════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and do a quick sanity-check."""
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path, dtype=str)          # read everything as str first
    log.info(f"Raw shape: {df.shape}")
    log.info(f"Columns   : {df.columns.tolist()}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  CLEAN NUMERIC / PRICE COLUMNS
# ═════════════════════════════════════════════════════════════════════════════

def _strip_currency(series: pd.Series) -> pd.Series:
    """Remove ₹, $, commas, spaces → float."""
    return (
        series
        .str.replace(r"[₹$,\s]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _strip_percent(series: pd.Series) -> pd.Series:
    """Remove % → float."""
    return (
        series
        .str.replace("%", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning numeric columns …")

    df["discounted_price"]    = _strip_currency(df["discounted_price"])
    df["actual_price"]        = _strip_currency(df["actual_price"])
    df["discount_percentage"] = _strip_percent(df["discount_percentage"])

    # rating: some rows contain '|' separated values – take the first
    df["rating"] = (
        df["rating"]
        .str.split("|").str[0]
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )

    # rating_count: remove commas  (e.g. "24,269" → 24269)
    df["rating_count"] = (
        df["rating_count"]
        .str.replace(",", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

    # Derived feature
    df["discount_amount"] = df["actual_price"] - df["discounted_price"]

    log.info("Numeric cleaning done.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EXPLODE MULTI-USER ROWS
# ═════════════════════════════════════════════════════════════════════════════

def explode_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row has comma-separated user_id AND review_id.
    Expand them so each row = one (user, product) interaction.
    """
    log.info("Exploding multi-user rows …")

    df = df.copy()

    # Split both lists
    df["user_id"]   = df["user_id"].str.split(",")
    df["review_id"] = df["review_id"].str.split(",")

    # Explode user_id
    df = df.explode("user_id").reset_index(drop=True)
    df["user_id"] = df["user_id"].str.strip()

    # Match review_id length to user_id (they should be parallel lists)
    # After exploding user_id the review_id column still holds a list;
    # we take the element at the same relative position.
    def _pick_review(row):
        """Return the review_id at the same index as the user in the original list."""
        lst = row["review_id"]
        if not isinstance(lst, list):
            return lst
        # We cannot know which index this user was without extra bookkeeping,
        # so we assign sequentially after the explode (index within group).
        return lst  # will be resolved below

    # Simpler: explode review_id independently and zip
    df_reviews = (
        df.groupby("product_id", sort=False)["review_id"]
        .apply(lambda x: [item for sublist in x for item in
                           (sublist if isinstance(sublist, list) else [sublist])])
        .reset_index()
    )
    # Just drop review_id for now — it's not needed for the recommender
    df = df.drop(columns=["review_id"], errors="ignore")

    log.info(f"After explode: {df.shape[0]} rows, {df['user_id'].nunique()} unique users.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DROP DUPLICATES & HANDLE NULLS
# ═════════════════════════════════════════════════════════════════════════════

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Removing duplicates and nulls …")

    before = len(df)
    # Keep the highest rating if a user reviewed the same product twice
    df = (
        df.sort_values("rating", ascending=False)
          .drop_duplicates(subset=["user_id", "product_id"])
          .reset_index(drop=True)
    )
    log.info(f"Dropped {before - len(df)} duplicate (user, product) pairs.")

    # Drop rows where rating is missing (required for CF)
    df = df.dropna(subset=["rating"]).reset_index(drop=True)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 5.  COLD-START FILTER
# ═════════════════════════════════════════════════════════════════════════════

def filter_cold_start(df: pd.DataFrame,
                      min_user: int = MIN_USER_INTERACTIONS,
                      min_item: int = MIN_ITEM_INTERACTIONS) -> pd.DataFrame:
    """
    Iteratively remove users and items that appear too rarely.
    Stops when the matrix is stable.
    """
    log.info("Filtering cold-start users / items …")
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]
        item_counts = df["product_id"].value_counts()
        df = df[df["product_id"].isin(item_counts[item_counts >= min_item].index)]

    log.info(
        f"After {iteration} pass(es): {len(df)} interactions, "
        f"{df['user_id'].nunique()} users, {df['product_id'].nunique()} items."
    )
    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ENCODE USER & ITEM IDs
# ═════════════════════════════════════════════════════════════════════════════

def encode_ids(df: pd.DataFrame):
    """Map string IDs → contiguous integers. Returns df + encoders."""
    log.info("Encoding user and item IDs …")

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["product_id"])

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()
    log.info(f"n_users={n_users}  n_items={n_items}")

    return df, user_enc, item_enc


# ═════════════════════════════════════════════════════════════════════════════
# 7.  USER–ITEM INTERACTION MATRIX
# ═════════════════════════════════════════════════════════════════════════════

def build_interaction_matrix(df: pd.DataFrame) -> sp.csr_matrix:
    """Sparse user × item matrix filled with ratings."""
    log.info("Building sparse interaction matrix …")
    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1

    matrix = sp.csr_matrix(
        (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    )
    density = matrix.nnz / (n_users * n_items) * 100
    log.info(f"Matrix shape: {matrix.shape}  |  density: {density:.4f}%")
    return matrix


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CONTENT FEATURES (for content-based / hybrid models)
# ═════════════════════════════════════════════════════════════════════════════

def build_content_features(df: pd.DataFrame):
    """
    Per-product feature dataframe:
      - TF-IDF on (product_name + about_product)
      - Normalised price & discount
      - One-hot top-level category
    """
    log.info("Building content features …")

    # One row per product
    product_df = df.drop_duplicates("product_id").set_index("product_id").copy()

    # ── Text ──────────────────────────────────────────────────────────────────
    product_df["text"] = (
        product_df["product_name"].fillna("") + " " +
        product_df["about_product"].fillna("")
    )
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(product_df["text"])  # sparse (n_items, vocab)

    # ── Numeric ───────────────────────────────────────────────────────────────
    numeric_cols = ["discounted_price", "actual_price",
                    "discount_percentage", "rating", "rating_count", "discount_amount"]
    num_df = product_df[numeric_cols].copy()
    num_df = num_df.fillna(num_df.median())

    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(num_df)

    # ── Category (top-level only) ─────────────────────────────────────────────
    # Categories look like "Electronics|Cables|USB"  – keep only the first token
    product_df["category_top"] = (
        product_df["category"]
        .fillna("Unknown")
        .str.split("|").str[0]
        .str.strip()
    )
    cat_dummies = pd.get_dummies(product_df["category_top"], prefix="cat")

    log.info(
        f"Content features – TF-IDF: {tfidf_matrix.shape}, "
        f"numeric: {num_scaled.shape}, "
        f"categories: {cat_dummies.shape[1]}"
    )
    return tfidf_matrix, tfidf, num_scaled, scaler, cat_dummies, product_df


# ═════════════════════════════════════════════════════════════════════════════
# 9.  TRAIN / TEST SPLIT  (leave-one-out — standard for recommenders)
# ═════════════════════════════════════════════════════════════════════════════

def leave_one_out_split(df: pd.DataFrame):
    """
    For each user, hold out their MOST RECENT interaction as the test item.
    Because we have no timestamps, we use the last row per user after sorting
    by rating (proxy for engagement quality).
    """
    log.info("Splitting train / test (leave-one-out) …")

    df = df.sort_values(["user_idx", "rating"], ascending=[True, False])
    test  = df.groupby("user_idx").tail(1)
    train = df.drop(test.index)

    log.info(f"Train: {len(train)} rows  |  Test: {len(test)} rows")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 10. SAVE ARTEFACTS
# ═════════════════════════════════════════════════════════════════════════════

def save_artefacts(
    train, test,
    interaction_matrix,
    tfidf_matrix, tfidf,
    num_scaled, scaler,
    cat_dummies,
    product_df,
    user_enc, item_enc,
    out_dir: str = PROCESSED_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Saving artefacts to  {out_dir}/")

    # DataFrames
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir,  "test.csv"),  index=False)
    product_df.reset_index().to_csv(os.path.join(out_dir, "products.csv"), index=False)

    # Sparse matrices
    sp.save_npz(os.path.join(out_dir, "interaction_matrix.npz"), interaction_matrix)
    sp.save_npz(os.path.join(out_dir, "tfidf_matrix.npz"),       tfidf_matrix)

    # Numeric array
    np.save(os.path.join(out_dir, "numeric_features.npy"), num_scaled)

    # Category dummies
    cat_dummies.reset_index().to_csv(os.path.join(out_dir, "category_dummies.csv"), index=False)

    # Sklearn objects
    joblib.dump(tfidf,    os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler,   os.path.join(out_dir, "numeric_scaler.pkl"))
    joblib.dump(user_enc, os.path.join(out_dir, "user_encoder.pkl"))
    joblib.dump(item_enc, os.path.join(out_dir, "item_encoder.pkl"))

    log.info("All artefacts saved ✓")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path: str = RAW_DATA_PATH):
    # 1. Load
    df = load_data(data_path)

    # 2. Clean numeric
    df = clean_numeric_columns(df)

    # 3. Explode multi-user rows
    df = explode_users(df)

    # 4. Deduplicate & drop nulls
    df = clean_interactions(df)

    # 5. Cold-start filter
    df = filter_cold_start(df)

    # 6. Encode IDs
    df, user_enc, item_enc = encode_ids(df)

    # 7. Interaction matrix
    interaction_matrix = build_interaction_matrix(df)

    # 8. Content features
    tfidf_matrix, tfidf, num_scaled, scaler, cat_dummies, product_df = \
        build_content_features(df)

    # 9. Train / test split
    train, test = leave_one_out_split(df)

    # 10. Save
    save_artefacts(
        train, test,
        interaction_matrix,
        tfidf_matrix, tfidf,
        num_scaled, scaler,
        cat_dummies,
        product_df,
        user_enc, item_enc,
    )

    log.info("=== Preprocessing complete ===")
    return {
        "train": train,
        "test":  test,
        "interaction_matrix": interaction_matrix,
        "tfidf_matrix": tfidf_matrix,
        "product_df": product_df,
        "user_enc": user_enc,
        "item_enc": item_enc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Amazon dataset")
    parser.add_argument("--data", default=RAW_DATA_PATH,
                        help="Path to raw amazon.csv")
    args = parser.parse_args()
    run_pipeline(args.data)