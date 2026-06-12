import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import recommender function
from src.content_based import recommend

# Load dataset
products = pd.read_csv("data/processed/products.csv")  # Update path as needed

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")

# Replace 'description' with your actual text column
tfidf_matrix = tfidf.fit_transform(products["about_product"].fillna(""))

# Streamlit UI
st.title("Product Recommendation System")

st.set_page_config(
    page_title="Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
# 🛍️Product Recommendation System
### AI-Powered Product Recommendation Engine
""")


algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Content Based",
        "Collaborative",
        "SVD",
        "Hybrid"
    ]
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Recommendations",
        "Evaluation Results"
    ]
)
product_name = st.text_input("Enter Product Name")

if st.button("Recommend"):

    recommendations = recommend(
        product_name,
        products,
        tfidf_matrix,
        top_n=5
    )

    st.subheader("Recommended Products")
    st.write(recommendations)
st.write(product_name)