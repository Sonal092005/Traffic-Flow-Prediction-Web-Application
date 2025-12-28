import pandas as pd
import numpy as np

np.random.seed(42)

data = []
for weekday in range(7):
    for hour in range(24):
        for junction in range(8):
            # Hyderabad has high traffic during peak hours, especially IT corridors
            if (6 <= hour <= 11 or 17 <= hour <= 21) and junction in [0, 1, 2, 3, 4]:
                base = 35  # High peak traffic for IT corridors
            elif 12 <= hour <= 16:
                base = 23  # Moderate afternoon
            elif hour >= 22 or hour <= 4:
                base = 7   # Low late night
            else:
                base = 16  # Off-peak
            
            # Add some variation
            intensity = max(1, min(40, int(np.random.normal(base, 5))))
            data.append([hour, weekday, junction, intensity])

df = pd.DataFrame(data, columns=['hour', 'weekday', 'junction', 'traffic_intensity'])
df.to_csv('data/hyderabad_traffic_dataset_lstm.csv', index=False)
print('Hyderabad dataset created with', len(df), 'records')
