import os
import pandas as pd
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medidas_2026_20")
numbered_files = [f"{i:03d}.txt" for i in range(1, 62)]

# Channel bins (256 bins each)
channels = {
    'Canal A (840-845 MHz)': (0, 256),
    'Canal B (845-850 MHz)': (256, 512),
    'Canal C (850-855 MHz)': (512, 768),
    'Canal D (855-860 MHz)': (768, 1024),
}

all_specs_raw = []
all_specs_clean = []
metadata_list = []

for f in numbered_files:
    path = os.path.join(folder, f)
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        parts = fp.readline().strip().split(',')
    
    spec_raw = np.array([float(x) for x in parts[:1024]])
    
    # Cleaning rule: "Cambia los datos menores a -65.0 por el valor de -95.0"
    spec_clean = np.where(spec_raw < -65.0, -95.0, spec_raw)
    
    all_specs_raw.append(spec_raw)
    all_specs_clean.append(spec_clean)
    
    metadata_list.append({
        'file': f,
        'temp': float(parts[1024]),
        'lon': float(parts[1025]),
        'lat': float(parts[1026]),
        'alt': float(parts[1027]),
        'err': float(parts[1028]),
    })

all_specs_raw = np.array(all_specs_raw)
all_specs_clean = np.array(all_specs_clean)

print("=== SPECTRUM ANALYSIS ACROSS ALL 61 MEASUREMENTS ===")
print("Shape of specs matrix:", all_specs_raw.shape)

# Let's check max power per bin across all measurements
max_per_bin_raw = np.max(all_specs_raw, axis=0)
max_per_bin_clean = np.max(all_specs_clean, axis=0)
mean_per_bin_clean = np.mean(all_specs_clean, axis=0)

freqs = np.linspace(840.0, 860.0, 1024, endpoint=False)

print(f"\nFrequency range: {freqs[0]:.3f} MHz to {freqs[-1]:.3f} MHz (step {freqs[1]-freqs[0]:.4f} MHz)")

# Channel level metrics
print("\n--- Channel Comparison (Raw vs Clean < -65 -> -95) ---")
for ch_name, (start_b, end_b) in channels.items():
    ch_raw = all_specs_raw[:, start_b:end_b]
    ch_clean = all_specs_clean[:, start_b:end_b]
    
    # Threshold condition: "Se considera un canal ocupado/contaminado aquel que tenga potencias superiores de -60 dBm de potencia"
    occupied_points_raw = np.sum(ch_raw > -60.0)
    occupied_points_clean = np.sum(ch_clean > -60.0)
    total_points = ch_raw.size
    
    # Parseval average power in linear (mW): P_lin = 10^(P_dBm/10)
    # Parseval sum: Sum P_lin / N
    p_lin_clean = 10.0 ** (ch_clean / 10.0)
    mean_power_mW = np.mean(p_lin_clean)
    mean_power_dBm = 10.0 * np.log10(mean_power_mW)
    
    # Direct dBm mean (sometimes used in literature / rough approximation)
    mean_dBm_direct = np.mean(ch_clean)
    
    max_dBm = np.max(ch_clean)
    
    print(f"\n{ch_name}:")
    print(f"  Max power: {max_dBm:.2f} dBm")
    print(f"  Points > -60 dBm: {occupied_points_clean} / {total_points} ({occupied_points_clean/total_points*100:.2f}%)")
    print(f"  Parseval Mean Power (linear converted to dBm): {mean_power_dBm:.2f} dBm")
    print(f"  Direct Mean dBm: {mean_dBm_direct:.2f} dBm")

# Find the most contaminated and least contaminated individual frequency
print("\n--- Individual Frequencies (1024 bins) ---")
# Count how many times each frequency is > -60 dBm
occ_per_freq = np.sum(all_specs_clean > -60.0, axis=0)
mean_lin_per_freq = 10.0 * np.log10(np.mean(10.0 ** (all_specs_clean / 10.0), axis=0))

most_occ_freq_idx = np.argmax(occ_per_freq)
least_occ_freq_idx = np.argmin(occ_per_freq)

most_pwr_freq_idx = np.argmax(mean_lin_per_freq)
least_pwr_freq_idx = np.argmin(mean_lin_per_freq)

print(f"Most contaminated by occupancy (> -60 dBm): {freqs[most_occ_freq_idx]:.4f} MHz (Bin {most_occ_freq_idx}) with {occ_per_freq[most_occ_freq_idx]} / 61 hits, mean power {mean_lin_per_freq[most_occ_freq_idx]:.2f} dBm")
print(f"Least contaminated by occupancy: {freqs[least_occ_freq_idx]:.4f} MHz (Bin {least_occ_freq_idx}) with {occ_per_freq[least_occ_freq_idx]} / 61 hits, mean power {mean_lin_per_freq[least_occ_freq_idx]:.2f} dBm")

print(f"Highest mean power frequency: {freqs[most_pwr_freq_idx]:.4f} MHz (Bin {most_pwr_freq_idx}) with mean power {mean_lin_per_freq[most_pwr_freq_idx]:.2f} dBm")
print(f"Lowest mean power frequency: {freqs[least_pwr_freq_idx]:.4f} MHz (Bin {least_pwr_freq_idx}) with mean power {mean_lin_per_freq[least_pwr_freq_idx]:.2f} dBm")
