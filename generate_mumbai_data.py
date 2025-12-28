import pandas as pd
import numpy as np

np.random.seed(42)

data = []
for weekday in range(7):
    for hour in range(24):
        for junction in range(8):
            # Mumbai has very high traffic during peak hours across most junctions
            if (6 <= hour <= 11 or 17 <= hour <= 21) and junction in [0, 1, 2, 3, 4, 5, 6]:
                base = 38  # Very high peak traffic
            elif 12 <= hour <= 16:
                base = 25  # Moderate afternoon
            elif hour >= 22 or hour <= 4:
                base = 8   # Low late night
            else:
                base = 18  # Off-peak
            
            # Add some variation
            intensity = max(1, min(40, int(np.random.normal(base, 5))))
            data.append([hour, weekday, junction, intensity])

df = pd.DataFrame(data, columns=['hour', 'weekday', 'junction', 'traffic_intensity'])
df.to_csv('data/mumbai_traffic_dataset_lstm.csv', index=False)
print('Mumbai dataset created with', len(df), 'records')
