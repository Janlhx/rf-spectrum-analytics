import os
import pandas as pd
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medidas_2026_20")
numbered_files = [f"{i:03d}.txt" for i in range(1, 62)]

rows = []
for f in numbered_files:
    path = os.path.join(folder, f)
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        line = fp.readline().strip()
    parts = line.split(',')
    spec = [float(x) for x in parts[:1024]]
    temp = float(parts[1024])
    lon = float(parts[1025])
    lat = float(parts[1026])
    alt = float(parts[1027])
    err = float(parts[1028])
    
    rows.append({
        'file': f,
        'temp': temp,
        'lon': lon,
        'lat': lat,
        'alt': alt,
        'err': err,
        'min_spec': min(spec),
        'max_spec': max(spec),
        'mean_spec': np.mean(spec),
        'under_minus_65': sum(1 for x in spec if x < -65.0)
    })

df_num = pd.DataFrame(rows)
print(df_num.to_string())
