
🚦 Traffic Flow Prediction Web Application

Live Demo: https://traffic-flow-prediction-web-application.onrender.com

A full-stack web application that forecasts traffic flow and congestion levels for major Indian cities using LSTM deep learning models. Designed for practical use, research, and intelligent decision-making.

🔍 Project Overview

Traffic congestion is a major challenge in urban environments, leading to delays, fuel waste, and environmental impact. This application uses time-series traffic data and deep learning techniques (LSTM) to predict future traffic flow based on user-provided date, time, and location.

It delivers:

- Numerical traffic predictions

- Categorical traffic level (Low / Moderate / High)

- Downloadable prediction results

- The app is deployed using Render with auto-deploy via GitHub.

⚙️ Live Application

🔗 Access the application here:
https://traffic-flow-prediction-web-application.onrender.com

Try it out by selecting:

- City

- Location

- Date & Time


Then submit to view predicted traffic flow.

🧠 Features

✔ Deep learning–based traffic prediction using LSTM
✔ Prediction of traffic level (Low / Moderate / High)
✔ Interactive UI with responsive design
✔ Download prediction data (CSV)
✔ Cloud deployment with auto-deploy
✔ Lightweight and scalable

🧰 Tech Stack

- Frontend

- HTML

- CSS

- JavaScript

- Backend

- Python

- Flask

- Gunicorn

- Machine Learning

- LSTM models (PyTorch)

- NumPy, Pandas

- Deployment

- GitHub (source control)

- Render (hosting & auto-deploy)

📁 Repository Structure
Traffic-Flow-Prediction-Web-Application/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── traffic_lstm_*.pth
│   ├── train_*.py
│
├── frontend/
│   ├── index.html
│   ├── main.css
│   └── scripts/
│       └── app.js
│
├── data/
│   └── *_traffic_dataset_lstm.csv
│
├── requirements.txt
├── README.md
└── .gitignore

🛠️ Local Setup (Optional)

If you want to run the project locally:

1. Clone the repository
   
git clone https://github.com/Sonal092005/Traffic-Flow-Prediction-Web-Application.git

cd Traffic-Flow-Prediction-Web-Application

3. Create and activate a virtual environment

python -m venv venv

On Windows
venv\Scripts\activate

On macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the application
python backend/app.py


Open your browser at:

http://127.0.0.1:5000/

🧠 Model Details

LSTM (Long Short-Term Memory) models are trained on historical traffic data for cities such as:

= Hyderabad

- Delhi

- Mumbai

- Pune

- Kolkata

- Kalaburagi

Models are stored in the backend/model/ directory and loaded dynamically by the Flask server.

📊 Usage Guide

- Open the app in your browser

- Select a city and location

- Choose date and time

- Click Predict

- View output — predicted traffic flow value & traffic level

Optional: Click Download Prediction Data to save results

# OUTPUT 

<img width="1832" height="885" alt="Screenshot 2025-12-28 174535" src="https://github.com/user-attachments/assets/5d2506f3-e445-477e-ade8-75003d4e1db3" />
<img width="1837" height="880" alt="Screenshot 2025-12-28 174550" src="https://github.com/user-attachments/assets/be887977-84cd-42d3-8d46-478b4c1d4b21" />
<img width="1824" height="881" alt="Screenshot 2025-12-28 174604" src="https://github.com/user-attachments/assets/db640ade-ab51-41c7-9e2d-91f1b5e2f1fb" />
<img width="1829" height="876" alt="Screenshot 2025-12-28 174622" src="https://github.com/user-attachments/assets/16ef4cf9-465a-4788-ab79-5ba7cc588f35" />
<img width="1816" height="875" alt="Screenshot 2025-12-28 174637" src="https://github.com/user-attachments/assets/c6989d0b-81af-42b1-ab1a-044a8431296b" />


📈 Future Improvements

🚀 Real-time traffic API integration
📍 Map-based visualization
📱 Mobile-responsive design tweaks
🔐 User accounts and saved prediction history
📊 Data charts & trend analytics

👩‍💻 Author

Sonal M Sangapur
B.E. – Information Science & Engineering
PDA College of Engineering, Kalaburagi

📧: sangapursonal@gmail.com

GitHub: https://github.com/Sonal092005

📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.
