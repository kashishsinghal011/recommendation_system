"""
ncf.py - Neural Collaborative Filtering (NCF)
==============================================
Combines GMF (Generalized Matrix Factorization) and MLP (Multi-Layer Perceptron)
into a NeuMF hybrid model for rating prediction.

Dataset: Amazon Product Reviews
Columns used: user_id, product_id, rating
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH   = "data/amazon.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "ncf_model.pt")

# Training hyper-parameters
EMBED_DIM   = 32        # embedding size for GMF & MLP branches
MLP_LAYERS  = [64, 32, 16]   # hidden layer sizes for MLP branch
DROPOUT     = 0.2
BATCH_SIZE  = 512
EPOCHS      = 20
LR          = 1e-3
NEG_SAMPLES = 4         # negative samples per positive interaction
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────

def load_and_encode(path: str):
    """
    Reads the Amazon CSV, cleans rating, and encodes user/product IDs
    to consecutive integer indices.
    """
    df = pd.read_csv(path, usecols=["user_id", "product_id", "rating"])

    # user_id column may contain comma-separated multiple users per row
    # Explode them so each row = one (user, product, rating)
    df["user_id"] = df["user_id"].astype(str).str.split(",")
    df = df.explode("user_id")
    df["user_id"] = df["user_id"].str.strip()

    # Clean rating – keep numeric values in [1, 5]
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
    print(f"[NCF] Users: {n_users} | Items: {n_items} | Interactions: {len(df)}")

    return df, n_users, n_items, user_enc, item_enc


def negative_sampling(df: pd.DataFrame, n_items: int, n_neg: int = NEG_SAMPLES):
    """
    For each positive (user, item) pair generate n_neg random negative items
    that the user has NOT interacted with.
    Returns a DataFrame with columns [user_idx, product_idx, label].
    """
    pos = df[["user_idx", "product_idx"]].copy()
    pos["label"] = 1.0

    user_item_set = set(zip(pos["user_idx"], pos["product_idx"]))
    neg_rows = []

    for user, item in zip(pos["user_idx"], pos["product_idx"]):
        sampled = 0
        while sampled < n_neg:
            neg_item = np.random.randint(0, n_items)
            if (user, neg_item) not in user_item_set:
                neg_rows.append((user, neg_item, 0.0))
                sampled += 1

    neg = pd.DataFrame(neg_rows, columns=["user_idx", "product_idx", "label"])
    combined = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=SEED)
    return combined


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users  = torch.tensor(df["user_idx"].values,    dtype=torch.long)
        self.items  = torch.tensor(df["product_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values,       dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ─── NeuMF Model ──────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (He et al., 2017).

    Architecture
    ────────────
      GMF branch : element-wise product of GMF embeddings  →  linear(embed_dim, 1)
      MLP branch : concat of MLP embeddings  →  MLP layers →  linear(mlp_layers[-1], 1)
      NeuMF      : concat(GMF output, MLP output)          →  sigmoid
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = EMBED_DIM,
                 mlp_layers: list = MLP_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()

        # ── GMF embeddings ────────────────────────────────────────────────────
        self.gmf_user_emb = nn.Embedding(n_users, embed_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embed_dim)

        # ── MLP embeddings ────────────────────────────────────────────────────
        self.mlp_user_emb = nn.Embedding(n_users, embed_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embed_dim)

        # ── MLP tower ─────────────────────────────────────────────────────────
        mlp_input_dim = embed_dim * 2
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # ── NeuMF output layer ────────────────────────────────────────────────
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.output.weight, a=1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        # GMF path
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i                         # element-wise product

        # MLP path
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_in  = torch.cat([mlp_u, mlp_i], dim=-1)    # concatenate
        mlp_out = self.mlp(mlp_in)

        # NeuMF fusion
        concat  = torch.cat([gmf_out, mlp_out], dim=-1)
        logit   = self.output(concat).squeeze(-1)
        return self.sigmoid(logit)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model: NeuMF, train_loader: DataLoader,
                val_loader: DataLoader, epochs: int = EPOCHS, lr: float = LR):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for users, items, labels in train_loader:
            users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(users, items)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
                preds    = model(users, items)
                val_loss += criterion(preds, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch:>3}/{epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model saved  (val_loss={best_val_loss:.4f})")

    print("\n[NCF] Training complete.")
    return model


# ─── Evaluation (Hit Rate @ K) ────────────────────────────────────────────────

def evaluate_hit_rate(model: NeuMF, test_df: pd.DataFrame,
                      n_items: int, k: int = 10) -> float:
    """
    Leave-one-out evaluation: for each test interaction, rank the true item
    against (k-1) random negatives.  Reports Hit Rate @ K.
    """
    model.eval()
    hits = 0

    for _, row in test_df.iterrows():
        user      = int(row["user_idx"])
        pos_item  = int(row["product_idx"])

        neg_items = np.random.choice(
            [i for i in range(n_items) if i != pos_item],
            size=k - 1, replace=False
        )

        all_items = np.concatenate([[pos_item], neg_items])
        user_t    = torch.tensor([user] * k, dtype=torch.long).to(DEVICE)
        item_t    = torch.tensor(all_items,  dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            scores = model(user_t, item_t).cpu().numpy()

        top_k_idx = np.argsort(-scores)[:k]
        if 0 in top_k_idx:      # index 0 is the positive item
            hits += 1

    hit_rate = hits / len(test_df)
    print(f"[NCF] Hit Rate @ {k}: {hit_rate:.4f}")
    return hit_rate


# ─── Inference ────────────────────────────────────────────────────────────────

def get_recommendations(model: NeuMF, user_id: str, user_enc: LabelEncoder,
                        item_enc: LabelEncoder, n_items: int,
                        top_n: int = 10) -> list:
    """
    Returns the top_n recommended product_ids for a given raw user_id string.
    """
    model.eval()

    try:
        user_idx = user_enc.transform([user_id])[0]
    except ValueError:
        print(f"[NCF] Unknown user: {user_id}")
        return []

    user_t = torch.tensor([user_idx] * n_items, dtype=torch.long).to(DEVICE)
    item_t = torch.arange(n_items, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        scores = model(user_t, item_t).cpu().numpy()

    top_indices   = np.argsort(-scores)[:top_n]
    product_ids   = item_enc.inverse_transform(top_indices)
    return list(zip(product_ids, scores[top_indices].round(4)))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   Neural Collaborative Filtering (NeuMF)")
    print("=" * 60)

    # 1. Load data
    df, n_users, n_items, user_enc, item_enc = load_and_encode(DATA_PATH)

    # 2. Negative sampling
    print("[NCF] Generating negative samples …")
    sampled_df = negative_sampling(df, n_items, n_neg=NEG_SAMPLES)

    # 3. Train / val / test split
    train_df, temp_df = train_test_split(sampled_df, test_size=0.2, random_state=SEED)
    val_df,   test_df = train_test_split(temp_df,    test_size=0.5, random_state=SEED)

    train_loader = DataLoader(InteractionDataset(train_df),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(InteractionDataset(val_df),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Build model
    model = NeuMF(n_users, n_items, EMBED_DIM, MLP_LAYERS, DROPOUT).to(DEVICE)
    print(f"[NCF] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Train
    model = train_model(model, train_loader, val_loader, EPOCHS, LR)

    # 6. Load best checkpoint and evaluate
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    pos_test = test_df[test_df["label"] == 1].reset_index(drop=True)
    evaluate_hit_rate(model, pos_test.head(500), n_items, k=10)

    # 7. Demo recommendation
    sample_user = df["user_id"].iloc[0]
    print(f"\n[NCF] Top-10 recommendations for user '{sample_user}':")
    recs = get_recommendations(model, sample_user, user_enc, item_enc, n_items)
    for rank, (pid, score) in enumerate(recs, 1):
        print(f"  {rank:>2}. {pid}  (score={score})")


if __name__ == "__main__":
    main()