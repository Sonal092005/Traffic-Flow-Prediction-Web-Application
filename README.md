
# Traffic Flow Prediction Using LSTM (Single Dataset Approach)

This project demonstrates how deep learning (LSTM) and machine learning can turn raw, time-series traffic data into actionable traffic flow predictions. Built as a modular, real-time web application, it is designed to aid urban congestion management and smart mobility use cases.

## 🔗 Live Demo
https://traffic-flow-prediction-web-application.onrender.com

## 🚀 Deployment
- Backend: Flask (Python)
- ML Model: LSTM (PyTorch)
- Web Server: Gunicorn
- Hosting Platform: Render
- Frontend: HTML, CSS, JavaScript
  
## ✨ Features

- Traffic flow prediction using an LSTM deep learning model
- Predicts hourly traffic volume for city junctions using historical sensor data
- LSTM neural network captures temporal patterns and traffic trends
- Real-time predictions exposed via a REST API
- Interactive web-based user interface
- Built using PyTorch, Flask, and modern web technologies
- Automated data preprocessing and repeatable model deployment
- Deployed online and publicly accessible

## Dataset

- Single time-series dataset (sample generated related to Bangalore Traffic)
- Includes: Hour, weekday, junction ID, and historic vehicle count

## Technologies Used

- **Python 3.x**
- **PyTorch** (deep learning)
- **Flask** (API & backend)
- **pandas, scikit-learn, joblib** (data cleaning, feature engineering, serialization)
- **HTML/CSS/JavaScript** (frontend)

## Project Structure
```
├── backend/
│   ├── app.py                         # Flask API
│   ├── data_preprocessing_and_training.py
│   └── model/
│       └── traffic_lstm.pth
├── frontend/
│   ├── index.html
│   ├── main.css
│   └── scripts/
│       └── app.js
├── data/
│   └── bangalore_traffic_dataset_lstm.csv
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository and install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Preprocess the data:
    ```
    python preprocess.py
    ```

3. Train or load the LSTM model:
    ```
    python train.py
    ```

4. Run the Flask API:
    ```
    python app.py
    ```

5. Access the web app via `localhost:5000` in your browser.

## Usage

- Enter hour, weekday, and (optionally) weather info via the web frontend.
- View predicted traffic flow instantly.

## Applications & Extensions

- Scalable for multiple junctions, bigger datasets, or extra features (holidays, accidents, etc.)
- Can deploy to the cloud or a local server

## OUTPUT

<img width="1827" height="884" alt="Screenshot 2025-12-27 155758" src="https://github.com/user-attachments/assets/df7b8e87-5fd9-41a1-8648-c955138bb830" />
<img width="1816" height="863" alt="Screenshot 2025-12-27 155831" src="https://github.com/user-attachments/assets/18c69ea8-e039-4831-b78a-33e7c63cf7c8" />
<img width="1827" height="855" alt="Screenshot 2025-12-27 155915" src="https://github.com/user-attachments/assets/426a0efa-43c0-4b67-bd4e-83d484767177" />
<img width="1600" height="850" alt="image" src="https://github.com/user-attachments/assets/0ee4a21b-0ac0-4fee-b182-1ef858f0ee14" />

## License

MIT License

Team Members :
    SALILA S PUNNESHETTY
    SONAL M SANGAPUR
    SWATI J SAJJAN

Information Science Engineering Student,
PDA College of Engineering
***

