
# Traffic Flow Prediction Using LSTM (Single Dataset Approach)

This project demonstrates how deep learning (LSTM) and machine learning can turn raw, time-series traffic data into actionable traffic flow predictions. Built as a modular, real-time web application, it is designed to aid urban congestion management and smart mobility use cases.

## Features

- Predicts hourly or daily traffic volume for a city junction using historic sensor data
- Built with PyTorch, Flask API, and a modern web frontend
- LSTM neural network captures temporal trends in traffic flow
- Automated data preprocessing and model deployment for repeatability

## Dataset

- Single time-series dataset (sample generated related to Banglore Traffic)
- Includes: Hour, weekday, weather info, and historic vehicle count per time interval

## Technologies Used

- **Python 3.x**
- **PyTorch** (deep learning)
- **Flask** (API & backend)
- **pandas, scikit-learn, joblib** (data cleaning, feature engineering, serialization)
- **HTML/CSS/JavaScript** (frontend)

## Project Structure

```
├── data/
│   └── traffic_data.csv
├── model/
│   └── lstm_model.pth
├── app.py            # Flask API
├── preprocess.py     # Data cleaning & feature engineering
├── static/           # Frontend static files
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

- Scalable for multiple junctions, bigger datasets, or extra features (weather, holidays, accidents, etc.)
- Can deploy to the cloud or a local server

## OUTPUT

<img width="1827" height="884" alt="Screenshot 2025-12-27 155758" src="https://github.com/user-attachments/assets/df7b8e87-5fd9-41a1-8648-c955138bb830" />



## License

MIT License

## Developed By
Salila and Team,
Information Science Engineering Student,
PDA College of Engineering

Team Members :
    SALILA S P
    SONAL M S
    SWATI S J

***

