import os
import pandas as pd
import numpy as np

# Load processed data
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida_procesada", "mediciones_procesadas_resumen.csv")
df = pd.read_csv(csv_path)

channels = ['Canal A', 'Canal B', 'Canal C', 'Canal D']

print("=== ESTIMACIÓN GEOGRÁFICA DE FUENTES DE CONTAMINACIÓN (BONIFICACIÓN) ===")

def estimate_source(df, power_col, alpha=2.0):
    # Linear power weight
    p_lin = 10.0 ** (df[power_col] / 10.0)
    # Give high weight to the top 20% highest measurements
    threshold = np.percentile(df[power_col], 75)
    weights = np.where(df[power_col] >= threshold, p_lin ** alpha, 0.0)
    if np.sum(weights) == 0:
        weights = p_lin
    
    weights /= np.sum(weights)
    lat_est = np.sum(weights * df['lat'])
    lon_est = np.sum(weights * df['lon'])
    
    # Peak measurement location
    idx_max = df[power_col].idxmax()
    row_max = df.loc[idx_max]
    
    return {
        'lat_est': lat_est,
        'lon_est': lon_est,
        'peak_lat': row_max['lat'],
        'peak_lon': row_max['lon'],
        'peak_file': row_max['archivo'],
        'peak_power': row_max[power_col]
    }

for ch in channels:
    res = estimate_source(df, f'{ch}_pot_parseval_dbm')
    print(f"\n{ch}:")
    print(f"  Pico Máximo en {res['peak_file']} ({res['peak_power']:.2f} dBm) en lat={res['peak_lat']:.6f}, lon={res['peak_lon']:.6f}")
    print(f"  Fuente Estimada (Centroide RF Ponderado): lat={res['lat_est']:.6f}, lon={res['lon_est']:.6f}")

# Also for most contaminated frequency (853.1445 MHz)
res_freq = estimate_source(df, 'pot_freq_mas_contaminada_dbm')
print(f"\nFrecuencia Más Contaminada (853.1445 MHz):")
print(f"  Pico Máximo en {res_freq['peak_file']} ({res_freq['peak_power']:.2f} dBm) en lat={res_freq['peak_lat']:.6f}, lon={res_freq['peak_lon']:.6f}")
print(f"  Fuente Estimada: lat={res_freq['lat_est']:.6f}, lon={res_freq['lon_est']:.6f}")
