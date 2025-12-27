import requests

data = {
    "hour": 8,
    "weekday": 1,
    "junction": 1
}
response = requests.post("http://127.0.0.1:5000/api/predict", json=data)
print(response.json())
