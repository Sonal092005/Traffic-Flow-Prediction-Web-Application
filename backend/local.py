import pandas as pd
import random

rows = []

for _ in range(8000):
    hour = random.randint(0, 23)
    weekday = random.randint(0, 6)
    junction = random.randint(0, 8)

    base = {0:30,1:25,2:28,3:24,4:22,5:26,6:35,7:23,8:20}[junction]

    if 7 <= hour <= 10 or 17 <= hour <= 21:
        time_factor = random.randint(5, 10)
    elif 11 <= hour <= 16:
        time_factor = random.randint(-2, 4)
    else:
        time_factor = random.randint(-8, -3)

    day_factor = random.randint(-5, 2) if weekday in [0,6] else random.randint(0,5)

    traffic = max(0, min(40, base + time_factor + day_factor))
    rows.append([hour, weekday, junction, traffic])

df = pd.DataFrame(rows, columns=[
    "hour", "weekday", "junction", "traffic_intensity"
])

df.to_csv("bangalore_traffic_dataset_lstm.csv", index=False)
