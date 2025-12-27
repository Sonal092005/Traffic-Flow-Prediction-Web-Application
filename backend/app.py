from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# -------------------------------
# Model directory
# -------------------------------
model_dir = os.path.join(os.path.dirname(__file__), "model")

scaler_X_path = os.path.join(model_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")

scaler_X = joblib.load(scaler_X_path) if os.path.exists(scaler_X_path) else None
scaler_y = joblib.load(scaler_y_path) if os.path.exists(scaler_y_path) else None

# -------------------------------
# LSTM Model Definition
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# -------------------------------
# Load model (ONLY valid file)
# -------------------------------
model_path = os.path.join(model_dir, "traffic_lstm.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError("traffic_lstm.pth not found")

model = LSTMModel(input_dim=3)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

LOOKBACK = 3

# -------------------------------
# Junction name mapping
# -------------------------------
JUNCTION_MAP = {
    0: "J0 – MG Road × Residency Road",
    1: "J1 – Outer Ring Road × Hebbal",
    2: "J2 – Silk Board Junction",
    3: "J3 – Whitefield Main Junction",
    4: "J4 – Manyata Tech Park Entrance",
    5: "J5 – Electronic City Signal",
    6: "J6 – Kengeri Bus Terminal Junction",
    7: "J7 – Yeshwanthpur Railway Junction",
    8: "J8 – Airport Road Trumpet Junction",
}

LOW_CAP_JUNCTIONS = {0, 1, 2}

# -------------------------------
# Flask App & Frontend
# -------------------------------
frontend_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend")
)

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(frontend_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(frontend_folder, path)

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}

    hour = int(data.get("hour", 0))
    weekday = int(data.get("weekday", 0))
    junction = int(data.get("junction", 0))

    if not (0 <= hour <= 23 and 0 <= weekday <= 6 and 0 <= junction <= 8):
        return jsonify({"error": "Invalid input range"}), 400

    # Prepare input
    X = np.array([[hour, weekday, junction]], dtype=float)

    if scaler_X is not None:
        X_scaled = scaler_X.transform(X)
    else:
        X_scaled = X

    X_seq = np.repeat(X_scaled[np.newaxis, :, :], LOOKBACK, axis=1)
    X_torch = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_torch).item()

    if scaler_y is not None:
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    else:
        pred = pred_scaled

    pred = max(0, min(40, pred))

    if junction in LOW_CAP_JUNCTIONS:
        pred = min(pred, 20)

    pred = round(pred)

    if pred == 0:
        level = "No Traffic"
    elif pred <= 20:
        level = "Low"
    elif pred <= 30:
        level = "Mild"
    elif pred <= 40:
        level = "High"
    else:
        level = "Peak"

    weekday_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    weekday_str = weekday_names[weekday]

    return jsonify({
        "predicted_traffic": float(pred),
        "traffic_level": level,
        "junction_id": junction,
        "junction_name": JUNCTION_MAP.get(junction),
        "hour": hour,
        "weekday": weekday,
        "time_label": f"{weekday_str} {hour:02d}:00"
    })

if __name__ == "__main__":
    app.run(debug=True)
