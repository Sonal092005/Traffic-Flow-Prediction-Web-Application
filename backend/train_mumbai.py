import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import random
from sklearn.preprocessing import MinMaxScaler
import os

# ===============================
# Reproducibility
# ===============================
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(__file__)            # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)        # project root
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mumbai_traffic_dataset_lstm.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Using CSV path:", DATA_PATH)

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv(DATA_PATH)

# ===============================
# Column Validation
# ===============================
required_cols = ["hour", "weekday", "junction", "traffic_intensity"]
missing = set(required_cols) - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ===============================
# Data Safety & Cleaning
# ===============================
data = data.dropna()
data = data[
    (data["hour"] >= 0) & (data["hour"] <= 23) &
    (data["weekday"] >= 0) & (data["weekday"] <= 6) &
    (data["junction"] >= 0) & (data["junction"] <= 7) &
    (data["traffic_intensity"] >= 0) & (data["traffic_intensity"] <= 40)
]

# Sort for temporal consistency
data = data.sort_values(by=["weekday", "hour", "junction"]).reset_index(drop=True)

# ===============================
# Feature & Target Selection
# ===============================
X = data[["hour", "weekday", "junction"]].values
y = data["traffic_intensity"].values.reshape(-1, 1)

# Avoid zero-range for scaler_y
if y.max() == y.min():
    y = y + np.random.normal(0, 0.01, size=y.shape)

# ===============================
# Scaling
# ===============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ===============================
# Sequence Creation
# ===============================
def create_sequences(X, y, lookback=3):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

LOOKBACK = 3
X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)

# ===============================
# Train-Test Split (time-ordered)
# ===============================
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).flatten()
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).flatten()

# ===============================
# LSTM Model
# ===============================
class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = TrafficLSTM(input_dim=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ===============================
# Training Loop
# ===============================
EPOCHS = 200
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_t).flatten()
    loss = criterion(predictions, y_train_t)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t).flatten()
            val_loss = criterion(val_pred, y_test_t)
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"
        )

# ===============================
# Quick sanity check on prediction range
# ===============================
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).flatten().numpy()
    test_pred = model(X_test_t).flatten().numpy()

print("Scaled train pred min/max:", train_pred.min(), train_pred.max())
print("Scaled test  pred min/max:", test_pred.min(), test_pred.max())
print("Scaled y_train min/max:", y_train_t.min().item(), y_train_t.max().item())
print("Scaled y_test  min/max:", y_test_t.min().item(), y_test_t.max().item())

# ===============================
# Save Model & Scalers for Mumbai
# ===============================
joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X_mumbai.pkl"))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y_mumbai.pkl"))
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "traffic_lstm_mumbai.pth"))

print("✅ Mumbai model training completed successfully.")
print("✅ Model files saved with '_mumbai' suffix.")
