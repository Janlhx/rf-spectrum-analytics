import os
import glob
import pandas as pd
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medidas_2026_20")
files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])

print(f"Total txt files found: {len(files)}")

data_records = []
errors = []

for f in files:
    path = os.path.join(folder, f)
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        lines = [line.strip() for line in fp.readlines() if line.strip()]
    
    if len(lines) == 0:
        errors.append((f, "Empty file"))
        continue
    
    # Check lines
    for line_idx, line in enumerate(lines):
        parts = line.split(',')
        n_parts = len(parts)
        
        # Check metadata
        if n_parts >= 1029:
            spec = parts[:1024]
            temp = parts[1024]
            lon = parts[1025]
            lat = parts[1026]
            alt = parts[1027]
            err = parts[1028]
            data_records.append({
                'file': f,
                'line': line_idx,
                'n_cols': n_parts,
                'temp': float(temp) if temp != '' else np.nan,
                'lon': float(lon) if lon != '' else np.nan,
                'lat': float(lat) if lat != '' else np.nan,
                'alt': float(alt) if alt != '' else np.nan,
                'err': float(err) if err != '' else np.nan,
                'min_spec': min([float(x) for x in spec]),
                'max_spec': max([float(x) for x in spec]),
            })
        else:
            errors.append((f, f"Incorrect column count: {n_parts}"))

df = pd.DataFrame(data_records)
print(f"Parsed {len(df)} measurements successfully.")
if errors:
    print(f"Errors found ({len(errors)}):")
    for e in errors:
        print("  ", e)

print("\n--- Summary of Metadata Columns ---")
print(df[['temp', 'lon', 'lat', 'alt', 'err', 'min_spec', 'max_spec']].describe())

print("\n--- Rows with (lon==0 or lat==0) or NaN ---")
zeros_or_null = df[(df['lon'] == 0) | (df['lat'] == 0) | df['lon'].isna() | df['lat'].isna()]
print(f"Count of invalid GPS points: {len(zeros_or_null)}")
for idx, r in zeros_or_null.iterrows():
    print(f"File: {r['file']}, Line: {r['line']}, lat: {r['lat']}, lon: {r['lon']}, alt: {r['alt']}, temp: {r['temp']}, err: {r['err']}")

print("\n--- High Error or Outlier check ---")
print("High distance error (err > 5 or err == 0):")
print(df[(df['err'] > 5) | (df['err'] <= 0)][['file', 'lon', 'lat', 'alt', 'err', 'temp']])

print("\n--- Latitude and Longitude Ranges (excluding 0) ---")
valid_coords = df[(df['lon'] != 0) & (df['lat'] != 0)]
print(f"Lat: {valid_coords['lat'].min()} to {valid_coords['lat'].max()}")
print(f"Lon: {valid_coords['lon'].min()} to {valid_coords['lon'].max()}")
print(f"Alt: {valid_coords['alt'].min()} to {valid_coords['alt'].max()}")
print(f"Temp: {valid_coords['temp'].min()} to {valid_coords['temp'].max()}")
