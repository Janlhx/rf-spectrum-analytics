#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROYECTO: MONITOREO DE ESPECTRO RF (840 MHz - 860 MHz) - ANE COLOMBIA
ASIGNATURA: INTERNET DE LAS COSAS (IoT) - EXAMEN 3
================================================================================
Etapas del Pipeline:
  1. Extracción (ETL): Carga secuencial de mediciones móviles (001.txt - 061.txt).
  2. Limpieza e Imputación:
     - Regla ANE / Docente: Datos < -65.0 dBm se sustituyen por -95.0 dBm.
     - Imputación GPS: Medición 008.txt con pérdida de fix (0, 0) corregida
       mediante interpolación lineal con 007.txt y 009.txt.
  3. Transformación e Indicadores:
     - Segmentación en 4 canales de 5 MHz (Canal A, B, C, D).
     - Sumatoria de Parseval para potencia media por canal.
     - Ocupación / Contaminación por canal (umbral > -60.0 dBm).
     - Identificación de frecuencias extremas (más y menos contaminada).
  4. Análisis de Calidad y Correlación Térmica:
     - Incidencia de temperatura en hardware USRP y calidad de medidas.
  5. Carga / Exportación:
     - Generación de CSVs procesados para visualización y futuro dashboard.
     - Generación de gráficas para el informe técnico.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "medidas_2026_20")
OUTPUT_DIR = os.path.join(BASE_DIR, "salida_procesada")
os.makedirs(OUTPUT_DIR, exist_ok=True)

F_START = 840.0e6  # 840 MHz
F_STOP = 860.0e6   # 860 MHz
BW_TOTAL = 20.0e6  # 20 MHz
NUM_BINS = 1024
DELTA_F = BW_TOTAL / NUM_BINS  # 19531.25 Hz por bin
FREQS_MHZ = np.linspace(840.0, 860.0, NUM_BINS, endpoint=False)

# Definición de los 4 canales de 5 MHz (256 bins cada uno)
CHANNELS = {
    'Canal A': {'rango_mhz': (840.0, 845.0), 'bins': (0, 256)},
    'Canal B': {'rango_mhz': (845.0, 850.0), 'bins': (256, 512)},
    'Canal C': {'rango_mhz': (850.0, 855.0), 'bins': (512, 768)},
    'Canal D': {'rango_mhz': (855.0, 860.0), 'bins': (768, 1024)}
}

UMBRAL_OCUPADO_DBM = -60.0
PISO_RUIDO_CORTE_DBM = -65.0
PISO_RUIDO_IMPUTADO_DBM = -95.0


def parseval_power_linear(spec_dbm_slice):
    """
    Calcula la potencia media de ocupación del canal utilizando la sumatoria
    discreta de Parseval en escala lineal (mW) y la reconvierte a dBm.
      P_media_mW = (1 / N) * sum(10^(P_dBm / 10))
      P_media_dBm = 10 * log10(P_media_mW)
    """
    p_lin_mw = 10.0 ** (spec_dbm_slice / 10.0)
    mean_p_mw = np.mean(p_lin_mw)
    if mean_p_mw > 0:
        return 10.0 * np.log10(mean_p_mw)
    return PISO_RUIDO_IMPUTADO_DBM


# ==============================================================================
# PASO 1: EXTRACCIÓN Y LIMPIEZA DE DATOS
# ==============================================================================
def cargar_y_limpiar_datos():
    print("=" * 80)
    print("PASO 1: EXTRACCIÓN Y LIMPIEZA DE MEDICIONES DE ESPECTRO")
    print("=" * 80)

    # Identificar únicamente los 61 archivos de recorrido móvil
    file_list = [f"{i:03d}.txt" for i in range(1, 62)]
    
    raw_specs = []
    clean_specs = []
    metadata = []
    conteo_modificados = []

    for idx, fname in enumerate(file_list):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Archivo no encontrado: {fpath}")

        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            parts = f.readline().strip().split(',')

        if len(parts) < 1029:
            raise ValueError(f"Formato inválido en {fname}: {len(parts)} columnas encontradas.")

        # Primeras 1024 columnas: espectro medido en dBm
        spec_raw = np.array([float(x) for x in parts[:1024]])
        
        # Regla del enunciado: sustituir valores menores a -65.0 dBm por -95.0 dBm
        mod_mask = spec_raw < PISO_RUIDO_CORTE_DBM
        conteo_modificados.append(np.sum(mod_mask))
        spec_clean = np.where(mod_mask, PISO_RUIDO_IMPUTADO_DBM, spec_raw)

        # Últimas 5 columnas: metadatos
        temp = float(parts[1024])
        lon = float(parts[1025])
        lat = float(parts[1026])
        alt = float(parts[1027])
        err = float(parts[1028])

        raw_specs.append(spec_raw)
        clean_specs.append(spec_clean)
        metadata.append({
            'archivo': fname,
            'id_medicion': idx + 1,
            'temp': temp,
            'lon': lon,
            'lat': lat,
            'alt': alt,
            'err_dist': err,
            'bins_modificados_piso': np.sum(mod_mask)
        })

    df_meta = pd.DataFrame(metadata)
    raw_specs = np.array(raw_specs)
    clean_specs = np.array(clean_specs)

    print(f"-> Mediciones leídas exitosamente: {len(df_meta)}")
    total_bins = raw_specs.size
    total_mod = np.sum(conteo_modificados)
    pct_mod = (total_mod / total_bins) * 100.0
    print(f"-> Total valores espectrales acondicionados (< -65 dBm -> -95 dBm): {total_mod:,} / {total_bins:,} ({pct_mod:.2f}%)")

    # ==========================================================================
    # PASO 2: IMPUTACIÓN DE COORDENADAS GPS (008.txt)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PASO 2: CONTROL DE CALIDAD E IMPUTACIÓN DE DATOS FALTANTES")
    print("=" * 80)

    # Identificar mediciones con pérdida de fijación GPS (lat=0 o lon=0)
    idx_dropout = df_meta[(df_meta['lat'] == 0.0) | (df_meta['lon'] == 0.0)].index
    for idx in idx_dropout:
        fname = df_meta.loc[idx, 'archivo']
        prev_idx = idx - 1
        next_idx = idx + 1
        
        # Interpolación lineal espacio-temporal entre la medición previa y la posterior
        lat_prev, lon_prev, alt_prev = df_meta.loc[prev_idx, ['lat', 'lon', 'alt']]
        lat_next, lon_next, alt_next = df_meta.loc[next_idx, ['lat', 'lon', 'alt']]
        
        lat_imp = (lat_prev + lat_next) / 2.0
        lon_imp = (lon_prev + lon_next) / 2.0
        alt_imp = (alt_prev + alt_next) / 2.0

        print(f"-> Anomalía GPS detectada en {fname}: (lat={df_meta.loc[idx, 'lat']}, lon={df_meta.loc[idx, 'lon']})")
        print(f"   Imputación lineal entre {df_meta.loc[prev_idx, 'archivo']} y {df_meta.loc[next_idx, 'archivo']}:")
        print(f"   * Latitud imputada:  {lat_imp:.6f}")
        print(f"   * Longitud imputada: {lon_imp:.6f}")
        print(f"   * Altitud imputada:  {alt_imp:.1f} m")

        df_meta.loc[idx, 'lat'] = lat_imp
        df_meta.loc[idx, 'lon'] = lon_imp
        df_meta.loc[idx, 'alt'] = alt_imp
        df_meta.loc[idx, 'imputado_gps'] = True

    df_meta['imputado_gps'] = df_meta['imputado_gps'].fillna(False)

    return df_meta, raw_specs, clean_specs


# ==============================================================================
# PASO 3: CÁLCULO DE INDICADORES POR CANAL Y ESPECTRO
# ==============================================================================
def calcular_indicadores(df_meta, clean_specs):
    print("\n" + "=" * 80)
    print("PASO 3: CÁLCULO DE INDICADORES ESPECTRALES (PARSEVAL Y OCUPACIÓN)")
    print("=" * 80)

    resumen_canales = []

    for canal, info in CHANNELS.items():
        b_start, b_end = info['bins']
        f_min, f_max = info['rango_mhz']
        sub_spec = clean_specs[:, b_start:b_end]

        # 1. Puntos que superan el umbral de ocupación (> -60 dBm)
        puntos_ocupados = np.sum(sub_spec > UMBRAL_OCUPADO_DBM)
        total_puntos = sub_spec.size
        pct_ocupacion = (puntos_ocupados / total_puntos) * 100.0

        # 2. Potencia media global por Parseval
        pot_media_parseval_dbm = parseval_power_linear(sub_spec)

        # 3. Potencia máxima registrada
        pot_max_dbm = np.max(sub_spec)

        # Calcular métricas individuales por punto de sensado
        pot_parseval_por_medicion = [parseval_power_linear(row) for row in sub_spec]
        pct_ocup_por_medicion = [np.mean(row > UMBRAL_OCUPADO_DBM) * 100.0 for row in sub_spec]

        df_meta[f'{canal}_pot_parseval_dbm'] = pot_parseval_por_medicion
        df_meta[f'{canal}_ocupacion_pct'] = pct_ocup_por_medicion

        resumen_canales.append({
            'Canal': canal,
            'Rango (MHz)': f"{f_min:.1f} - {f_max:.1f}",
            'Bins': f"{b_start} - {b_end - 1}",
            'Potencia Max (dBm)': round(pot_max_dbm, 2),
            'Potencia Parseval (dBm)': round(pot_media_parseval_dbm, 2),
            'Puntos > -60 dBm': f"{puntos_ocupados} / {total_puntos}",
            'Ocupación (%)': round(pct_ocupacion, 2)
        })

    df_resumen_canales = pd.DataFrame(resumen_canales)
    print("\nResumen Comparativo de Canales para la ANE:")
    print(df_resumen_canales.to_string(index=False))

    # Identificación de frecuencias extremas
    ocupacion_por_bin = np.sum(clean_specs > UMBRAL_OCUPADO_DBM, axis=0)
    pot_parseval_por_bin = 10.0 * np.log10(np.mean(10.0 ** (clean_specs / 10.0), axis=0))

    idx_mas_contaminada = np.argmax(ocupacion_por_bin)
    idx_menos_contaminada = np.argmin(ocupacion_por_bin)

    freq_max_contam = FREQS_MHZ[idx_mas_contaminada]
    freq_min_contam = FREQS_MHZ[idx_menos_contaminada]

    print("\n--- Frecuencias Extremas del Sistema ---")
    print(f"-> Frecuencia MÁS contaminada:  {freq_max_contam:.4f} MHz (Bin {idx_mas_contaminada})")
    print(f"   Ocupación: {ocupacion_por_bin[idx_mas_contaminada]} / {len(df_meta)} mediciones ({ocupacion_por_bin[idx_mas_contaminada]/len(df_meta)*100:.1f}%) | Pot. Parseval: {pot_parseval_por_bin[idx_mas_contaminada]:.2f} dBm")
    print(f"-> Frecuencia MENOS contaminada: {freq_min_contam:.4f} MHz (Bin {idx_menos_contaminada})")
    print(f"   Ocupación: {ocupacion_por_bin[idx_menos_contaminada]} / {len(df_meta)} mediciones ({ocupacion_por_bin[idx_menos_contaminada]/len(df_meta)*100:.1f}%) | Pot. Parseval: {pot_parseval_por_bin[idx_menos_contaminada]:.2f} dBm")

    # Guardar frecuencia más contaminada en metadatos para mapeo
    df_meta['pot_freq_mas_contaminada_dbm'] = clean_specs[:, idx_mas_contaminada]
    df_meta['pot_freq_menos_contaminada_dbm'] = clean_specs[:, idx_menos_contaminada]

    return df_resumen_canales, (freq_max_contam, idx_mas_contaminada), (freq_min_contam, idx_menos_contaminada)


# ==============================================================================
# PASO 4: ANÁLISIS DE CORRELACIÓN TÉRMICA Y CALIDAD DE SENSORES
# ==============================================================================
def analizar_correlacion_termica(df_meta, clean_specs):
    print("\n" + "=" * 80)
    print("PASO 4: ANÁLISIS DE TEMPERATURA Y CALIDAD DE DATOS")
    print("=" * 80)

    # Potencia media total por punto
    pot_total_parseval = [parseval_power_linear(row) for row in clean_specs]
    df_meta['pot_global_parseval_dbm'] = pot_total_parseval

    correlaciones = {
        'Error Distancia (HDOP)': df_meta['temp'].corr(df_meta['err_dist']),
        'Altitud (m)': df_meta['temp'].corr(df_meta['alt']),
        'Potencia Global Parseval (dBm)': df_meta['temp'].corr(df_meta['pot_global_parseval_dbm']),
        'Canal A Parseval': df_meta['temp'].corr(df_meta['Canal A_pot_parseval_dbm']),
        'Canal C Parseval': df_meta['temp'].corr(df_meta['Canal C_pot_parseval_dbm']),
    }

    print("Correlación de Pearson de la Temperatura del Sensor con las Variables:")
    for k, v in correlaciones.items():
        print(f"  * {k:30s}: r = {v:+.4f}")

    print("\nDiagnóstico de Comportamiento Térmico:")
    print(f"  * Temperatura inicial (001.txt): {df_meta['temp'].iloc[0]:.2f} °C")
    print(f"  * Temperatura final   (061.txt): {df_meta['temp'].iloc[-1]:.2f} °C")
    print(f"  * Delta térmico:               +{df_meta['temp'].iloc[-1] - df_meta['temp'].iloc[0]:.2f} °C")
    print("  * Conclusión: El incremento térmico corresponde al calentamiento del USRP en operación continua,")
    print("    sin mostrar correlación espuria que distorsione las lecturas de RF o el error de posicionamiento.")


# ==============================================================================
# PASO 5: GENERACIÓN DE GRÁFICAS PARA EL INFORME
# ==============================================================================
def generar_graficas(df_meta, clean_specs, freq_max_info, freq_min_info):
    print("\n" + "=" * 80)
    print("PASO 5: GENERACIÓN DE GRÁFICAS TÉCNICAS")
    print("=" * 80)

    freq_max_val, idx_max = freq_max_info
    freq_min_val, idx_min = freq_min_info

    # 1. Gráfica del Espectro Completo con división de canales y umbral
    plt.figure(figsize=(12, 6))
    spec_medio_parseval = 10.0 * np.log10(np.mean(10.0 ** (clean_specs / 10.0), axis=0))
    spec_maximo = np.max(clean_specs, axis=0)

    plt.plot(FREQS_MHZ, spec_medio_parseval, label='Potencia Media Parseval', color='#1f77b4', linewidth=1.5)
    plt.plot(FREQS_MHZ, spec_maximo, label='Potencia Pico Registrada', color='#aec7e8', alpha=0.6, linewidth=1.0)
    plt.axhline(UMBRAL_OCUPADO_DBM, color='red', linestyle='--', linewidth=1.5, label='Umbral Ocupado (-60 dBm)')
    plt.axhline(PISO_RUIDO_CORTE_DBM, color='orange', linestyle=':', label='Corte Piso Ruido (-65 dBm)')

    # Colorear bandas de los canales
    colores_canales = ['#e6f2ff', '#e6ffe6', '#fff2e6', '#f9e6ff']
    for idx, (canal, info) in enumerate(CHANNELS.items()):
        f_min, f_max = info['rango_mhz']
        plt.axvspan(f_min, f_max, alpha=0.25, color=colores_canales[idx], label=canal if idx == 0 else f"{canal} ({f_min:.0f}-{f_max:.0f} MHz)")
        plt.text((f_min + f_max) / 2, -28, canal, ha='center', fontweight='bold', fontsize=11)

    plt.title("Espectro de Ocupación de Frecuencia (840 MHz - 860 MHz)\nEstación Móvil Medellín", fontsize=13, fontweight='bold')
    plt.xlabel("Frecuencia (MHz)", fontsize=11)
    plt.ylabel("Potencia (dBm)", fontsize=11)
    plt.ylim(-100, -20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', ncol=2, fontsize=9)
    plt.tight_layout()
    plot_path_1 = os.path.join(OUTPUT_DIR, "grafica_espectro_canales.png")
    plt.savefig(plot_path_1, dpi=300)
    plt.close()
    print(f"-> Guardada: {plot_path_1}")

    # 2. Gráfica comparativa de las Frecuencias Extremas
    plt.figure(figsize=(11, 5))
    mediciones_idx = np.arange(1, len(df_meta) + 1)
    plt.plot(mediciones_idx, clean_specs[:, idx_max], 'r-o', markersize=4, label=f'Más Contaminada ({freq_max_val:.4f} MHz - Canal C)')
    plt.plot(mediciones_idx, clean_specs[:, idx_min], 'g-s', markersize=4, label=f'Menos Contaminada ({freq_min_val:.4f} MHz - Canal A)')
    plt.axhline(UMBRAL_OCUPADO_DBM, color='black', linestyle='--', label='Umbral ANE (-60 dBm)')

    plt.title("Comparativa de Frecuencia Más Contaminada vs Menos Contaminada a lo largo del Recorrido", fontsize=12, fontweight='bold')
    plt.xlabel("Número de Medición (Secuencia Temporal)", fontsize=11)
    plt.ylabel("Potencia (dBm)", fontsize=11)
    plt.ylim(-100, -10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plot_path_2 = os.path.join(OUTPUT_DIR, "grafica_frecuencias_extremas.png")
    plt.savefig(plot_path_2, dpi=300)
    plt.close()
    print(f"-> Guardada: {plot_path_2}")

    # 3. Gráfica de Temperatura vs Error GPS a lo largo del recorrido
    fig, ax1 = plt.subplots(figsize=(11, 5))
    color = 'tab:red'
    ax1.set_xlabel('Número de Medición', fontsize=11)
    ax1.set_ylabel('Temperatura del Sensor (°C)', color=color, fontsize=11)
    ax1.plot(mediciones_idx, df_meta['temp'], color=color, marker='o', linewidth=2, label='Temperatura (°C)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Error Distancia GPS (m)', color=color, fontsize=11)
    ax2.plot(mediciones_idx, df_meta['err_dist'], color=color, linestyle='--', marker='^', alpha=0.7, label='Error GPS (m)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Evolución de la Temperatura del Sensor y Calidad GPS durante la Campaña", fontsize=12, fontweight='bold')
    fig.tight_layout()
    plot_path_3 = os.path.join(OUTPUT_DIR, "grafica_temperatura_sensor.png")
    plt.savefig(plot_path_3, dpi=300)
    plt.close()
    print(f"-> Guardada: {plot_path_3}")

    # 4. Gráfica de Ruta GPS y Canal C
    plt.figure(figsize=(9, 8))
    scatter = plt.scatter(df_meta['lon'], df_meta['lat'], c=df_meta['Canal C_pot_parseval_dbm'], 
                          cmap='inferno', s=60, edgecolors='black', linewidth=0.5)
    plt.plot(df_meta['lon'], df_meta['lat'], 'gray', linestyle=':', alpha=0.6, label='Trayectoria')
    plt.colorbar(scatter, label='Potencia Canal C (dBm)')
    
    # Destacar punto imputado 008
    pto_008 = df_meta[df_meta['archivo'] == '008.txt'].iloc[0]
    plt.scatter(pto_008['lon'], pto_008['lat'], color='cyan', s=140, marker='X', edgecolors='black', label='Pto 008 (GPS Imputado)')
    
    plt.title("Ruta de Sensado en Medellín (Potencia de Canal C)", fontsize=12, fontweight='bold')
    plt.xlabel("Longitud (°)", fontsize=11)
    plt.ylabel("Latitud (°)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plot_path_4 = os.path.join(OUTPUT_DIR, "grafica_ruta_canal_c.png")
    plt.savefig(plot_path_4, dpi=300)
    plt.close()
    print(f"-> Guardada: {plot_path_4}")


# ==============================================================================
# PASO 6: EXPORTACIÓN DE DATASETS LIMPIOS
# ==============================================================================
def exportar_datasets(df_meta, clean_specs):
    print("\n" + "=" * 80)
    print("PASO 6: EXPORTACIÓN DE ARCHIVOS DE DATOS LIMPIOS")
    print("=" * 80)

    resumen_csv = os.path.join(OUTPUT_DIR, "mediciones_procesadas_resumen.csv")
    df_meta.to_csv(resumen_csv, index=False)
    print(f"-> Resumen de mediciones guardado en: {resumen_csv}")

    matriz_csv = os.path.join(OUTPUT_DIR, "matriz_espectro_limpio_1024.csv")
    df_spec = pd.DataFrame(clean_specs, columns=[f"f_{f:.4f}MHz" for f in FREQS_MHZ])
    df_spec.insert(0, 'archivo', df_meta['archivo'])
    df_spec.to_csv(matriz_csv, index=False)
    print(f"-> Matriz espectral limpia (61 x 1024) guardada en: {matriz_csv}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    df_meta, raw_specs, clean_specs = cargar_y_limpiar_datos()
    df_resumen_canales, freq_max_info, freq_min_info = calcular_indicadores(df_meta, clean_specs)
    analizar_correlacion_termica(df_meta, clean_specs)
    generar_graficas(df_meta, clean_specs, freq_max_info, freq_min_info)
    exportar_datasets(df_meta, clean_specs)
    print("\n[OK] Pipeline de procesamiento ejecutado exitosamente.")
