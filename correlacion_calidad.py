import os
import pandas as pd
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medidas_2026_20")
numbered_files = [f"{i:03d}.txt" for i in range(1, 62)]

rows = []
for f in numbered_files:
    path = os.path.join(folder, f)
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        parts = fp.readline().strip().split(',')
    
    spec_raw = np.array([float(x) for x in parts[:1024]])
    spec_clean = np.where(spec_raw < -65.0, -95.0, spec_raw)
    
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
        'raw_mean': np.mean(spec_raw),
        'raw_min': np.min(spec_raw),
        'raw_max': np.max(spec_raw),
        'clean_mean': np.mean(spec_clean),
        'num_clean_modified': np.sum(spec_raw < -65.0),
        'num_occupied': np.sum(spec_clean > -60.0),
        'parseval_mean_lin_dBm': 10 * np.log10(np.mean(10**(spec_clean/10)))
    })

df = pd.DataFrame(rows)

print("=== CORRELATION ANALYSIS (Sensors & Temperature) ===")
# Exclude 008 with 0,0 for GPS correlations
valid_gps = df[df['lat'] > 0]

corr_cols = ['temp', 'err', 'alt', 'raw_mean', 'raw_min', 'clean_mean', 'num_occupied', 'num_clean_modified', 'parseval_mean_lin_dBm']
corr_matrix = df[corr_cols].corr()
print("\nCorrelation with Temperature:")
print(corr_matrix['temp'].sort_values(ascending=False))

print("\nCorrelation with GPS Error:")
print(corr_matrix['err'].sort_values(ascending=False))

print("\n--- Imputation summary for cleaning rule (< -65 dBm -> -95.0 dBm) ---")
total_values = 61 * 1024
total_modified = df['num_clean_modified'].sum()
print(f"Total spectral values: {total_values}")
print(f"Total spectral values < -65 dBm modified to -95.0 dBm: {total_modified} ({total_modified / total_values * 100:.2f}%)")
print(f"Per-file average modified: {df['num_clean_modified'].mean():.1f} bins ({df['num_clean_modified'].mean()/1024*100:.2f}%)")

print("\n--- Temperature Progression ---")
print(f"Starting Temp (001.txt): {df.iloc[0]['temp']:.2f} °C")
print(f"Ending Temp (061.txt): {df.iloc[-1]['temp']:.2f} °C")
print(f"Min Temp: {df['temp'].min():.2f} °C, Max Temp: {df['temp'].max():.2f} °C")
