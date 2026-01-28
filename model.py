# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# RankNet (RegressionNet)
# =============================================================================
class RegressionNet(nn.Module):
    def __init__(self, d, hidden=128, depth=5, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = d
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

def train_ranknet(X_tr, Y_tr, epochs=300, lr=0.003, hidden=128, depth=5, batch_size=128, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [RankNet] Training on {device}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_tr, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(X_train_t) > batch_size)
    )

    model = RegressionNet(d=X_train.shape[1], hidden=hidden, depth=depth, dropout=0.15).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, Y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_weights)
    return (model, scaler)

def predict_ranknet(loaded_obj, X):
    model, scaler = loaded_obj
    device = next(model.parameters()).device
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        scores = model(X_t).cpu().numpy().flatten()
    return scores, None