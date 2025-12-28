from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# -------------------------------
# Model directory and registry
# -------------------------------
model_dir = os.path.join(os.path.dirname(__file__), "model")

MODEL_REGISTRY = {
    "bangalore": {
        "model": os.path.join(model_dir, "traffic_lstm.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y.pkl"),
    },
    "kalaburagi": {
        "model": os.path.join(model_dir, "traffic_lstm_kalaburagi.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_kalaburagi.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_kalaburagi.pkl"),
    },
    "mumbai": {
        "model": os.path.join(model_dir, "traffic_lstm_mumbai.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_mumbai.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_mumbai.pkl"),
    },
    "hyderabad": {
        "model": os.path.join(model_dir, "traffic_lstm_hyderabad.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_hyderabad.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_hyderabad.pkl"),
    },
    "pune": {
        "model": os.path.join(model_dir, "traffic_lstm_pune.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_pune.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_pune.pkl"),
    },
    "delhi": {
        "model": os.path.join(model_dir, "traffic_lstm_delhi.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_delhi.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_delhi.pkl"),
    },
    "kolkata": {
        "model": os.path.join(model_dir, "traffic_lstm_kolkata.pth"),
        "scaler_X": os.path.join(model_dir, "scaler_X_kolkata.pkl"),
        "scaler_y": os.path.join(model_dir, "scaler_y_kolkata.pkl"),
    },
}

MODEL_CACHE = {}

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
# Model loading helper
# -------------------------------

def load_city_assets(city: str):
    if city not in MODEL_REGISTRY:
        return None

    if city in MODEL_CACHE:
        return MODEL_CACHE[city]

    paths = MODEL_REGISTRY[city]
    model_path = paths["model"]

    if not os.path.exists(model_path):
        return None

    model = LSTMModel(input_dim=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler_X = joblib.load(paths["scaler_X"]) if os.path.exists(paths["scaler_X"]) else None
    scaler_y = joblib.load(paths["scaler_y"]) if os.path.exists(paths["scaler_y"]) else None

    MODEL_CACHE[city] = (model, scaler_X, scaler_y)
    return MODEL_CACHE[city]

LOOKBACK = 3

# -------------------------------
# City-specific configuration
# -------------------------------
CITY_CONFIG = {
    "bangalore": {
        "junction_map": {
            0: "Silk Board",
            1: "KR Puram",
            2: "Marathahalli ORR",
            3: "Hebbal",
            4: "Electronic City",
            5: "Whitefield ITPL",
            6: "Yeshwanthpur",
            7: "Kengeri",
        },
        "high_bias": {0, 1, 2, 3, 4, 5},
        "low_cap": {7},
    },
    "kalaburagi": {
        "junction_map": {
            0: "Timapuri",
            1: "Ram Mandir Circle",
            2: "Jagat Circle",
            3: "Super Market",
            4: "Kharge",
            5: "Aland Check Post",
            6: "Humnabad Ring Road",
            7: "University",
        },
        "high_bias": {0, 1, 2, 3},
        "low_cap": set(),
    },
    "mumbai": {
        "junction_map": {
            0: "Dadar TT Circle",
            1: "Bandra Reclamation",
            2: "Andheri Subway",
            3: "Powai Junction",
            4: "BKC Connector",
            5: "Lower Parel Flyover",
            6: "Worli Sea Link",
            7: "CST Junction",
        },
        "high_bias": {0, 1, 2, 3, 4, 5, 6},
        "low_cap": set(),
    },
    "hyderabad": {
        "junction_map": {
            0: "HITEC City",
            1: "Gachibowli Junction",
            2: "Madhapur Circle",
            3: "Ameerpet Metro",
            4: "Begumpet Junction",
            5: "Secunderabad Circle",
            6: "LB Nagar Junction",
            7: "Uppal Ring Road",
        },
        "high_bias": {0, 1, 2, 3, 4},
        "low_cap": set(),
    },
    "pune": {
        "junction_map": {
            0: "Hinjewadi IT Park",
            1: "Wakad Junction",
            2: "Aundh Circle",
            3: "Deccan Gymkhana",
            4: "Shivaji Nagar",
            5: "Hadapsar Junction",
            6: "Kothrud Depot",
            7: "Swargate Junction",
        },
        "high_bias": {0, 1, 2, 3, 4, 5},
        "low_cap": set(),
    },
    "delhi": {
        "junction_map": {
            0: "ITO Crossing",
            1: "Connaught Place",
            2: "Dhaula Kuan",
            3: "Kashmere Gate",
            4: "Nehru Place",
            5: "Rajiv Chowk",
            6: "Akshardham Junction",
            7: "Dwarka Sector 21",
        },
        "high_bias": {0, 1, 2, 3, 4, 5, 6, 7},
        "low_cap": set(),
    },
    "kolkata": {
        "junction_map": {
            0: "Park Street Crossing",
            1: "Esplanade",
            2: "Sealdah Junction",
            3: "Howrah Bridge",
            4: "Salt Lake Bypass",
            5: "Gariahat Crossing",
            6: "EM Bypass Connector",
            7: "Maidan Junction",
        },
        "high_bias": {0, 1, 2, 3, 4, 5},
        "low_cap": set(),
    },
}

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

    city = str(data.get("city", "bangalore")).strip().lower()
    config = CITY_CONFIG.get(city)
    if config is None:
        return jsonify({"error": "Unsupported city. Try 'bangalore', 'kalaburagi', 'mumbai', 'hyderabad', 'pune', 'delhi', or 'kolkata'."}), 400

    junction_map = config["junction_map"]
    max_junction = max(junction_map.keys())

    hour = int(data.get("hour", 0))
    weekday = int(data.get("weekday", 0))
    junction = int(data.get("junction", 0))

    if not (0 <= hour <= 23 and 0 <= weekday <= 6 and 0 <= junction <= max_junction):
        return jsonify({"error": "Invalid input range"}), 400

    assets = load_city_assets(city)
    if assets is None:
        return jsonify({
            "error": (
                f"Model assets for {city} not found. "
                "Train and place model/scaler files in backend/model/."
            )
        }), 500

    model, scaler_X, scaler_y = assets

    # Prepare input
    X = np.array([[hour, weekday, junction]], dtype=float)
    X_scaled = scaler_X.transform(X) if scaler_X is not None else X

    X_seq = np.repeat(X_scaled[np.newaxis, :, :], LOOKBACK, axis=1)
    X_torch = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_torch).item()

    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0] if scaler_y is not None else pred_scaled
    pred = max(0, min(40, pred))

    # City-specific traffic patterns
    if city == "kalaburagi" and junction == 0:  # Timapuri
        if weekday == 0:  # Sunday - Low traffic all day
            pred = min(pred, 10)
        elif weekday >= 1 and weekday <= 5:  # Mon-Sat
            if 8 <= hour <= 9:  # 8:30 AM to 9:30 AM - High
                pred = max(pred, 28)
            elif 10 <= hour <= 12:  # 9:30 AM to 1 PM - Moderate
                pred = max(15, min(pred, 20))
            elif 13 <= hour <= 14:  # 1 PM to 2:30 PM - High
                pred = max(pred, 26)
            elif 15 <= hour <= 20:  # 2:30 PM to 9 PM - Moderate
                pred = max(15, min(pred, 20))
            else:  # 9 PM to 8:30 AM - Low
                pred = min(pred, 10)
    
    if city == "kalaburagi" and junction == 1:  # Ram Mandir Circle
        if weekday == 0:  # Sunday - Moderate traffic all day
            pred = max(15, min(pred, 20))
        elif 9 <= hour <= 12:  # 9 AM to 1 PM - High
            pred = max(pred, 26)
        elif 13 <= hour <= 14:  # 1 PM to 3 PM - Moderate
            pred = max(15, min(pred, 20))
        elif 15 <= hour <= 21:  # 3 PM to 10 PM - High
            pred = max(pred, 26)
        else:  # 10 PM to 9 AM - Low
            pred = min(pred, 10)
    
    if city == "kalaburagi" and junction == 2:  # Jagat Circle
        if weekday == 0:  # Sunday - Low traffic all day
            pred = min(pred, 10)
        elif hour == 9:  # 9 AM to 10 AM - High
            pred = max(pred, 26)
        elif 10 <= hour <= 15:  # 10 AM to 4 PM - Moderate
            pred = max(15, min(pred, 20))
        elif hour == 16:  # 4 PM to 5 PM - High
            pred = max(pred, 26)
        else:  # 5 PM to 9 AM - Moderate
            pred = max(15, min(pred, 20))
    
    if city == "kalaburagi" and junction == 3:  # Super Market
        if weekday == 0:  # Sunday - Moderate traffic all day
            pred = max(15, min(pred, 20))
        elif 12 <= hour <= 19:  # 12 PM to 8 PM - High
            pred = max(pred, 26)
        elif 20 <= hour <= 23:  # 8 PM to 12 AM - Moderate
            pred = max(15, min(pred, 20))
        else:  # 12 AM to 12 PM - Low
            pred = min(pred, 10)
    
    if city == "kalaburagi" and junction == 4:  # Kharge
        if weekday == 0:  # Sunday - Moderate traffic all day
            pred = max(15, min(pred, 20))
        elif 9 <= hour <= 20:  # 9 AM to 9 PM - High
            pred = max(pred, 26)
        elif 21 <= hour <= 23:  # 9 PM to 12 AM - Moderate
            pred = max(15, min(pred, 20))
        else:  # 12 AM to 9 AM - Low
            pred = min(pred, 10)
    
    if city == "kalaburagi" and junction == 5:  # Aland Check Post
        if weekday == 0:  # Sunday - Low traffic all day
            pred = min(pred, 10)
        elif 11 <= hour <= 17:  # 11 AM to 6 PM - High
            pred = max(pred, 26)
        elif 18 <= hour <= 23:  # 6 PM to 12 AM - Moderate
            pred = max(15, min(pred, 20))
        elif 0 <= hour <= 7:  # 12 AM to 8 AM - Low
            pred = min(pred, 10)
        else:  # 8 AM to 11 AM - Moderate
            pred = max(15, min(pred, 20))
    
    if city == "kalaburagi" and junction == 6:  # Humnabad Ring Road
        if weekday == 0:  # Sunday - Moderate traffic all day
            pred = max(15, min(pred, 20))
        elif 10 <= hour <= 21:  # 10 AM to 10 PM - High
            pred = max(pred, 26)
        elif 22 <= hour <= 23:  # 10 PM to 12 AM - Moderate
            pred = max(15, min(pred, 20))
        elif 0 <= hour <= 6:  # 12 AM to 7 AM - Low
            pred = min(pred, 10)
        else:  # 7 AM to 10 AM - Moderate
            pred = max(15, min(pred, 20))
    
    if city == "kalaburagi" and junction == 7:  # University
        if weekday == 0:  # Sunday - Moderate traffic all day
            pred = max(15, min(pred, 20))
        elif 9 <= hour <= 20:  # 9 AM to 9 PM - High
            pred = max(pred, 26)
        elif 21 <= hour <= 23:  # 9 PM to 12 AM - Moderate
            pred = max(15, min(pred, 20))
        else:  # 12 AM to 9 AM - Low
            pred = min(pred, 10)

    # Domain heuristics to bias results by time/weekday
    # Do not override explicit Kalaburagi Sunday rules requested by user
    if not (city == "kalaburagi" and weekday == 0):
        if weekday == 0 and 6 <= hour <= 11:  # Sunday morning outings
            pred = max(pred, 30)
        elif 6 <= hour <= 11:  # Weekday mornings are busy
            pred = max(pred, 25)
        elif 17 <= hour <= 21:  # Post-work/early night rush
            pred = max(pred, 25)
        elif hour >= 22 or hour <= 4:  # Late night
            pred = min(pred, 12)
        elif 12 <= hour <= 16:  # Afternoon tends to be moderate
            pred = max(pred, 18)

    high_bias = config.get("high_bias", set())
    low_cap = config.get("low_cap", set())

    # Avoid forcing High bias on Kalaburagi Sundays per user rules
    if junction in high_bias and (6 <= hour <= 11 or 17 <= hour <= 21):
        if not (city == "kalaburagi" and weekday == 0):
            pred = max(pred, 30)

    if junction in low_cap:
        pred = min(pred, 20)

    pred = round(pred)

    # Simplified traffic levels
    if pred <= 10:
        level = "Low Traffic"
    elif pred <= 20:
        level = "Moderate Traffic"
    else:
        level = "High Traffic"

    weekday_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    weekday_str = weekday_names[weekday]

    return jsonify({
        "predicted_traffic": float(pred),
        "traffic_level": level,
        "junction_id": junction,
        "junction_name": junction_map.get(junction),
        "city": city.title(),
        "hour": hour,
        "weekday": weekday,
        "time_label": f"{weekday_str} {hour:02d}:00"
    })

if __name__ == "__main__":
    app.run(debug=True)
