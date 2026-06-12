"""
content_based.py
----------------
Content-based product recommender for the Amazon dataset.

Uses pre-computed TF-IDF matrix (product text features) and cosine
similarity to find products most similar to a queried product.

Usage:
    python src/content_based.py                         # interactive prompt
    python src/content_based.py --product "USB Cable"  # direct query
    python src/content_based.py --product "USB Cable" --top 10
"""

import os
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR     = os.path.join("data", "processed")
PRODUCTS_PATH     = os.path.join(PROCESSED_DIR, "products.csv")
TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "tfidf_matrix.npz")
TFIDF_VEC_PATH    = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_artifacts() -> tuple[pd.DataFrame, sp.csr_matrix]:
    """
    Load products metadata and the pre-computed TF-IDF matrix.

    Returns
    -------
    products : pd.DataFrame
        One row per product with columns product_id, product_name, category, …
    tfidf_matrix : scipy.sparse.csr_matrix
        Shape (n_products, n_vocab) — row i corresponds to products.iloc[i].

    Raises
    ------
    FileNotFoundError
        If any required file is missing from data/processed/.
    """
    for path in [PRODUCTS_PATH, TFIDF_MATRIX_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run preprocessing.py first to generate processed data."
            )

    log.info(f"Loading products from  {PRODUCTS_PATH}")
    products = pd.read_csv(PRODUCTS_PATH)
    products = products.reset_index(drop=True)           # ensure 0-based index

    log.info(f"Loading TF-IDF matrix from  {TFIDF_MATRIX_PATH}")
    tfidf_matrix = sp.load_npz(TFIDF_MATRIX_PATH)

    log.info(
        f"Loaded {len(products)} products  |  "
        f"TF-IDF shape: {tfidf_matrix.shape}"
    )
    return products, tfidf_matrix


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PRODUCT LOOKUP  (partial / case-insensitive search)
# ═════════════════════════════════════════════════════════════════════════════

def find_product(query: str, products: pd.DataFrame) -> Optional[int]:
    """
    Find the DataFrame row-index of the best matching product.

    Matching strategy (in order):
      1. Exact match (case-insensitive)
      2. Partial substring match — returns the shortest name that contains
         the query (closest match heuristic)

    Parameters
    ----------
    query    : user-supplied product name / keyword
    products : product metadata DataFrame

    Returns
    -------
    int  — row index of the matched product, or None if nothing found.
    """
    if not isinstance(query, str) or not query.strip():
        log.error("Invalid query: must be a non-empty string.")
        return None

    query_lower = query.strip().lower()
    names_lower = products["product_name"].str.lower()

    # 1. Exact match
    exact = products[names_lower == query_lower]
    if not exact.empty:
        log.info(f"Exact match found: '{exact.iloc[0]['product_name']}'")
        return exact.index[0]

    # 2. Partial match — pick shortest name (most specific match)
    partial = products[names_lower.str.contains(query_lower, na=False)]
    if not partial.empty:
        best_idx = partial["product_name"].str.len().idxmin()
        log.info(f"Partial match found: '{products.loc[best_idx, 'product_name']}'")
        return best_idx

    log.warning(f"No product matched query: '{query}'")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 3.  COSINE SIMILARITY  (row-wise, on demand)
# ═════════════════════════════════════════════════════════════════════════════

def compute_similarity_scores(
    product_idx: int,
    tfidf_matrix: sp.csr_matrix,
) -> np.ndarray:
    """
    Compute cosine similarity between the queried product and all others.

    We compute only the single row we need rather than the full n×n matrix,
    keeping memory usage O(n) instead of O(n²).

    Parameters
    ----------
    product_idx  : row index of the query product
    tfidf_matrix : sparse TF-IDF feature matrix

    Returns
    -------
    np.ndarray of shape (n_products,) with similarity scores in [0, 1].
    """
    query_vec = tfidf_matrix[product_idx]                # shape (1, vocab)
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return scores


# ═════════════════════════════════════════════════════════════════════════════
# 4.  RECOMMEND
# ═════════════════════════════════════════════════════════════════════════════

def recommend(
    product_name: str,
    products: pd.DataFrame,
    tfidf_matrix: sp.csr_matrix,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the top-N most similar products for a given product name.

    Parameters
    ----------
    product_name : query string (partial or full product name)
    products     : product metadata DataFrame
    tfidf_matrix : pre-computed TF-IDF matrix (one row per product)
    top_n        : number of recommendations to return (default 5)

    Returns
    -------
    pd.DataFrame with columns:
        rank, product_id, product_name, category,
        discounted_price, similarity_score
    Returns an empty DataFrame if the product is not found.

    Raises
    ------
    ValueError
        If top_n is not a positive integer.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got: {top_n}")

    # ── Find query product ────────────────────────────────────────────────────
    idx = find_product(product_name, products)
    if idx is None:
        log.error(f"Product not found: '{product_name}'")
        return pd.DataFrame()

    matched_name = products.loc[idx, "product_name"]
    log.info(f"Generating {top_n} recommendations for: '{matched_name}'")

    # ── Similarity scores ─────────────────────────────────────────────────────
    scores = compute_similarity_scores(idx, tfidf_matrix)

    # ── Rank and exclude the query product itself ─────────────────────────────
    ranked_indices = np.argsort(scores)[::-1]             # descending
    ranked_indices = [i for i in ranked_indices if i != idx]
    top_indices    = ranked_indices[:top_n]

    # ── Build result DataFrame ────────────────────────────────────────────────
    result = products.loc[top_indices].copy()
    result["similarity_score"] = scores[top_indices].round(4)
    result["rank"] = range(1, len(result) + 1)

    display_cols = ["rank", "product_id", "product_name",
                    "category", "discounted_price", "similarity_score"]
    display_cols = [c for c in display_cols if c in result.columns]

    return result[display_cols].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PRETTY PRINT
# ═════════════════════════════════════════════════════════════════════════════

def print_recommendations(recs: pd.DataFrame, query: str) -> None:
    """Print recommendations in the required output format."""
    if recs.empty:
        print(f"\nNo recommendations found for: '{query}'")
        return

    print(f"\nRecommended Products for '{query}':\n")
    for _, row in recs.iterrows():
        name = row["product_name"]
        cat  = row.get("category", "")
        sim  = row.get("similarity_score", "")
        price = row.get("discounted_price", "")

        # Primary line: rank + name
        print(f"{int(row['rank'])}. {name}")

        # Optional detail line (category, price, similarity)
        details = []
        if pd.notna(cat)   and cat:   details.append(f"Category: {cat}")
        if pd.notna(price) and price: details.append(f"Price: ₹{price}")
        if sim != "":                  details.append(f"Similarity: {sim:.4f}")
        if details:
            print(f"   {'  |  '.join(details)}")

    print()


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(product_name: Optional[str] = None, top_n: int = 5) -> None:
    """
    Entry point — loads data, accepts a product query, prints recommendations.

    Parameters
    ----------
    product_name : optional; if None the user is prompted interactively.
    top_n        : number of recommendations to display.
    """
    # Load artefacts once
    try:
        products, tfidf_matrix = load_artifacts()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    # Interactive or CLI input
    if not product_name:
        product_name = input("Enter product name to get recommendations: ").strip()

    if not product_name:
        log.error("No product name provided. Exiting.")
        return

    # Get recommendations
    try:
        recs = recommend(product_name, products, tfidf_matrix, top_n=top_n)
    except ValueError as e:
        log.error(f"Invalid argument: {e}")
        return

    # Display
    print_recommendations(recs, product_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content-based product recommender")
    parser.add_argument("--product", type=str,  default=None, help="Product name to query")
    parser.add_argument("--top",     type=int,  default=5,    help="Number of recommendations")
    args = parser.parse_args()

    main(product_name=args.product, top_n=args.top)