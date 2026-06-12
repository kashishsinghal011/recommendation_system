# 🛍️ Recommender System

An end-to-end Machine Learning based Recommendation System built using Python. This project implements multiple recommendation techniques including Popularity-Based Filtering, Content-Based Filtering, Collaborative Filtering, Matrix Factorization (SVD), Neural Collaborative Filtering (NCF), and Hybrid Recommendation Models. The system is designed using the Amazon Product Dataset and provides personalized product recommendations through an interactive Streamlit web application.

---

## 🚀 Features

* Data preprocessing and feature engineering
* Popularity-based recommendations
* Content-based recommendations using TF-IDF and cosine similarity
* User-Based Collaborative Filtering
* Item-Based Collaborative Filtering
* Matrix Factorization using SVD
* Neural Collaborative Filtering (NCF) using PyTorch
* Hybrid Recommendation Engine
* Comprehensive model evaluation
* Interactive Streamlit dashboard
* GitHub-ready project structure

---

## 📂 Project Structure

```text
recommender_system/
│
├── app/
│   └── app.py
│
├── data/
│   ├── amazon.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── products.csv
│       ├── tfidf_matrix.npz
│       ├── interaction_matrix.npz
│       └── ...
│
├── models/
│   ├── ncf_model.pt
│   └── saved_models/
│
├── src/
│   ├── preprocessing.py
│   ├── popularity.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── svd.py
│   ├── ncf.py
│   ├── hybrid.py
│   ├── evaluation.py
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses the Amazon Product Recommendation Dataset containing:

* Product Information
* Product Categories
* Ratings and Reviews
* User IDs
* Product Descriptions
* Review Metadata

### Dataset Statistics

* 1,465 Products
* 9,000+ Users
* 11,000+ User-Item Interactions

---

## ⚙️ Technologies Used

### Programming Language

* Python 3

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* SciPy
* PyTorch
* Streamlit
* Matplotlib
* Joblib

---

## 🧠 Recommendation Models

### 1. Popularity-Based Recommendation

Recommends products based on overall popularity and rating frequency.

**Advantages**

* Fast
* No training required
* Works for new users

---

### 2. Content-Based Filtering

Uses:

* TF-IDF Vectorization
* Product Descriptions
* Product Categories
* Cosine Similarity

Recommends products similar to the selected product.

---

### 3. Collaborative Filtering

#### User-Based CF

Finds users with similar preferences and recommends products they liked.

#### Item-Based CF

Finds products that are frequently liked together.

---

### 4. Singular Value Decomposition (SVD)

Matrix factorization approach that learns latent user and item features.

Benefits:

* Handles sparse data
* Improved recommendation quality
* Captures hidden patterns

---

### 5. Neural Collaborative Filtering (NCF)

Deep learning based recommender using PyTorch.

Features:

* User Embeddings
* Item Embeddings
* Multi-Layer Perceptron
* Non-linear preference learning

---

### 6. Hybrid Recommendation System

Combines:

* Content-Based Filtering
* Collaborative Filtering
* SVD
* NCF

to improve recommendation accuracy and robustness.

---

## 🔄 Workflow

### Step 1: Data Preprocessing

```bash
python src/preprocessing.py
```

Tasks:

* Data cleaning
* Missing value handling
* Feature extraction
* User-item matrix generation
* Train/Test split

---

### Step 2: Train Individual Models

```bash
python src/popularity.py
python src/content_based.py
python src/collaborative.py
python src/svd.py
python src/ncf.py
```

---

### Step 3: Evaluate Models

```bash
python src/evaluation.py
```

Evaluation Metrics:

* RMSE
* MAE
* Precision@K
* Recall@K
* Hit Rate
* NDCG
* MRR
* Coverage
* Diversity

---

### Step 4: Run Streamlit Application

```bash
streamlit run app/app.py
```

---

## 📈 Evaluation Metrics

The following metrics are used for comparison:

| Metric      | Description                                     |
| ----------- | ----------------------------------------------- |
| RMSE        | Root Mean Squared Error                         |
| MAE         | Mean Absolute Error                             |
| Precision@K | Relevance of top recommendations                |
| Recall@K    | Coverage of relevant items                      |
| Hit Rate    | Whether at least one recommendation is relevant |
| NDCG        | Ranking quality                                 |
| MRR         | Mean Reciprocal Rank                            |
| Coverage    | Catalog coverage                                |
| Diversity   | Recommendation diversity                        |

---

## 🌐 Streamlit Dashboard

The web application provides:

* Product Search
* Personalized Recommendations
* Multiple Recommendation Algorithms
* Evaluation Results Visualization
* User-Friendly Interface

---

## 🎯 Future Improvements

* Real-time recommendation updates
* Transformer-based recommendation models
* Session-based recommendations
* Explainable AI recommendations
* Deployment on AWS/GCP
* Docker containerization
* Recommendation API service

---

## 🏆 Learning Outcomes

Through this project, the following concepts were explored:

* Machine Learning Pipelines
* Recommendation Systems
* Natural Language Processing
* Deep Learning with PyTorch
* Matrix Factorization
* Feature Engineering
* Model Evaluation
* Streamlit Deployment
* Software Engineering Practices

---

## 👨‍💻 Author

Developed as a complete Recommendation System project demonstrating classical, machine learning, and deep learning recommendation approaches on real-world Amazon product data.

⭐ If you found this project useful, consider giving it a star on GitHub!
