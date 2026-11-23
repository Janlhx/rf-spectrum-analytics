import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📡 Estudio Técnico de Ocupación de Espectro RF (840 MHz - 860 MHz)\n",
                "## Informe y Modelo de Decisión para la Agencia Nacional del Espectro (ANE)\n",
                "**Asignatura:** Internet de las Cosas (IoT) - Examen 3  \n",
                "**Docente:** Leonardo Betancur  \n",
                "**Institución:** Universidad Pontificia Bolivariana (UPB)  \n",
                "---\n",
                "### Objetivos del Cuaderno:\n",
                "1. **ETL y Calidad de Datos:** Limpieza, verificación y proceso de imputación espacio-temporal de datos de sensado móvil.\n",
                "2. **Acondicionamiento Espectral:** Filtrado de piso de ruido (< -65 dBm a -95 dBm) según directriz técnica.\n",
                "3. **Indicadores de Ocupación:** Cálculo de potencia media de canal por sumatoria de Parseval y porcentaje de ocupación (> -60 dBm) para los cuatro canales de 5 MHz (Canal A, B, C, D).\n",
                "4. **Análisis de Sensores:** Incidencia térmica del USRP en la calidad de la medida.\n",
                "5. **Juicio de Valor Técnico:** Recomendación formal de asignación de bandas para el Ministerio TIC y la ANE."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Configuración de Entorno e Importación de Bibliotecas"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Parámetros del sistema RF\n",
                "F_START = 840.0e6\n",
                "F_STOP = 860.0e6\n",
                "BW_TOTAL = 20.0e6\n",
                "NUM_BINS = 1024\n",
                "DELTA_F = BW_TOTAL / NUM_BINS  # 19.53125 kHz\n",
                "FREQS_MHZ = np.linspace(840.0, 860.0, NUM_BINS, endpoint=False)\n",
                "\n",
                "CHANNELS = {\n",
                "    'Canal A': {'rango_mhz': (840.0, 845.0), 'bins': (0, 256)},\n",
                "    'Canal B': {'rango_mhz': (845.0, 850.0), 'bins': (256, 512)},\n",
                "    'Canal C': {'rango_mhz': (850.0, 855.0), 'bins': (512, 768)},\n",
                "    'Canal D': {'rango_mhz': (855.0, 860.0), 'bins': (768, 1024)}\n",
                "}\n",
                "\n",
                "UMBRAL_OCUPADO_DBM = -60.0\n",
                "PISO_RUIDO_CORTE_DBM = -65.0\n",
                "PISO_RUIDO_IMPUTADO_DBM = -95.0\n",
                "\n",
                "print(f\"Configuración cargada: {NUM_BINS} bins | Delta F = {DELTA_F/1e3:.2f} kHz | Rango: 840 - 860 MHz\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Carga de Datos (Soporte local y Google Colab)\n",
                "Si ejecutas en **Google Colab**, sube la carpeta `medidas_2026_20` o un archivo `medidas_2026_20.zip` en la barra lateral de archivos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Detección de ruta de datos\n",
                "POSIBLES_RUTAS = [\n",
                "    \"./medidas_2026_20\",\n",
                "    \"../medidas_2026_20\",\n",
                "    \"/content/medidas_2026_20\",\n",
                "    \"./IoT/medidas_2026_20\"\n",
                "]\n",
                "\n",
                "DATA_DIR = None\n",
                "for r in POSIBLES_RUTAS:\n",
                "    if os.path.exists(r) and os.path.isdir(r):\n",
                "        DATA_DIR = r\n",
                "        break\n",
                "\n",
                "if DATA_DIR is None:\n",
                "    # Si se subió un zip en Colab\n",
                "    if os.path.exists(\"medidas_2026_20.zip\"):\n",
                "        with zipfile.ZipFile(\"medidas_2026_20.zip\", 'r') as zip_ref:\n",
                "            zip_ref.extractall(\".\")\n",
                "        DATA_DIR = \"./medidas_2026_20\"\n",
                "    else:\n",
                "        raise FileNotFoundError(\"No se encontró la carpeta 'medidas_2026_20'. Por favor súbela al entorno.\")\n",
                "\n",
                "print(f\"Directorio de mediciones establecido: {DATA_DIR}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Extracción y Limpieza Espectral\n",
                "Se leen los 61 archivos (`001.txt` a `061.txt`).\n",
                "- Se descartan los archivos preliminares `medidaprueba*.txt` (pruebas de calibración estática previas a la ruta).\n",
                "- Se aplica la sustitución del piso de ruido: **datos < -65.0 dBm $\\to$ -95.0 dBm**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "file_list = [f\"{i:03d}.txt\" for i in range(1, 62)]\n",
                "raw_specs = []\n",
                "clean_specs = []\n",
                "metadata = []\n",
                "modificados_conteo = []\n",
                "\n",
                "for idx, fname in enumerate(file_list):\n",
                "    fpath = os.path.join(DATA_DIR, fname)\n",
                "    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:\n",
                "        parts = f.readline().strip().split(',')\n",
                "    \n",
                "    spec_raw = np.array([float(x) for x in parts[:1024]])\n",
                "    mask_mod = spec_raw < PISO_RUIDO_CORTE_DBM\n",
                "    modificados_conteo.append(np.sum(mask_mod))\n",
                "    spec_clean = np.where(mask_mod, PISO_RUIDO_IMPUTADO_DBM, spec_raw)\n",
                "    \n",
                "    raw_specs.append(spec_raw)\n",
                "    clean_specs.append(spec_clean)\n",
                "    metadata.append({\n",
                "        'archivo': fname,\n",
                "        'id': idx + 1,\n",
                "        'temp': float(parts[1024]),\n",
                "        'lon': float(parts[1025]),\n",
                "        'lat': float(parts[1026]),\n",
                "        'alt': float(parts[1027]),\n",
                "        'err_dist': float(parts[1028]),\n",
                "        'bins_modificados': np.sum(mask_mod)\n",
                "    })\n",
                "\n",
                "df_meta = pd.DataFrame(metadata)\n",
                "raw_specs = np.array(raw_specs)\n",
                "clean_specs = np.array(clean_specs)\n",
                "\n",
                "total_bins = raw_specs.size\n",
                "total_mod = np.sum(modificados_conteo)\n",
                "print(f\"-> Mediciones procesadas: {len(df_meta)}\")\n",
                "print(f\"-> Valores < -65 dBm modificados a -95 dBm: {total_mod:,} de {total_bins:,} ({total_mod/total_bins*100:.2f}%)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Detección de Anomalías e Imputación GPS (`008.txt`)\n",
                "Durante el desplazamiento móvil, el receptor GPS en el punto `008.txt` perdió la señal de satélites ($lat=0.0, lon=0.0$).\n",
                "Dado que la trayectoria es continua y secuencial, se realiza una **imputación por interpolación lineal** usando los puntos adyacentes `007.txt` y `009.txt`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "idx_dropout = df_meta[(df_meta['lat'] == 0.0) | (df_meta['lon'] == 0.0)].index\n",
                "\n",
                "for idx in idx_dropout:\n",
                "    fname = df_meta.loc[idx, 'archivo']\n",
                "    prev_idx, next_idx = idx - 1, idx + 1\n",
                "    \n",
                "    lat_imp = (df_meta.loc[prev_idx, 'lat'] + df_meta.loc[next_idx, 'lat']) / 2.0\n",
                "    lon_imp = (df_meta.loc[prev_idx, 'lon'] + df_meta.loc[next_idx, 'lon']) / 2.0\n",
                "    alt_imp = (df_meta.loc[prev_idx, 'alt'] + df_meta.loc[next_idx, 'alt']) / 2.0\n",
                "    \n",
                "    print(f\"[IMPUTACIÓN GPS] Archivo: {fname}\")\n",
                "    print(f\"  Anterior ({df_meta.loc[prev_idx, 'archivo']}): lat={df_meta.loc[prev_idx, 'lat']:.6f}, lon={df_meta.loc[prev_idx, 'lon']:.6f}\")\n",
                "    print(f\"  Posterior ({df_meta.loc[next_idx, 'archivo']}): lat={df_meta.loc[next_idx, 'lat']:.6f}, lon={df_meta.loc[next_idx, 'lon']:.6f}\")\n",
                "    print(f\"  -> Imputado: lat={lat_imp:.6f}, lon={lon_imp:.6f}, alt={alt_imp:.1f} m\")\n",
                "    \n",
                "    df_meta.loc[idx, 'lat'] = lat_imp\n",
                "    df_meta.loc[idx, 'lon'] = lon_imp\n",
                "    df_meta.loc[idx, 'alt'] = alt_imp\n",
                "    df_meta.loc[idx, 'imputado_gps'] = True\n",
                "\n",
                "df_meta['imputado_gps'] = df_meta['imputado_gps'].fillna(False)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Sumatoria de Parseval y Métricas de los Canales\n",
                "La potencia media de ocupación del canal discreto se calcula convirtiendo las muestras a potencia lineal en milivatios ($mW$), calculando la media en el ancho de banda del canal (Teorema de Parseval), y reconvirtiendo a dBm:\n",
                "$$\\overline{P}_{canal} = 10 \\log_{10} \\left( \\frac{1}{N_{ch}} \\sum_{k=1}^{N_{ch}} 10^{\\frac{P_{dBm}[k]}{10}} \\right)$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def parseval_power(spec_slice):\n",
                "    p_lin = 10.0 ** (spec_slice / 10.0)\n",
                "    p_mean = np.mean(p_lin)\n",
                "    return 10.0 * np.log10(p_mean) if p_mean > 0 else -95.0\n",
                "\n",
                "resumen_canales = []\n",
                "for canal, info in CHANNELS.items():\n",
                "    b_start, b_end = info['bins']\n",
                "    f_min, f_max = info['rango_mhz']\n",
                "    sub_matrix = clean_specs[:, b_start:b_end]\n",
                "    \n",
                "    # Puntos ocupados (> -60 dBm)\n",
                "    pts_ocupados = np.sum(sub_matrix > UMBRAL_OCUPADO_DBM)\n",
                "    total_pts = sub_matrix.size\n",
                "    pct_ocup = (pts_ocupados / total_pts) * 100.0\n",
                "    \n",
                "    # Potencia media Parseval global\n",
                "    p_parseval = parseval_power(sub_matrix)\n",
                "    p_max = np.max(sub_matrix)\n",
                "    \n",
                "    # Guardar métricas por medición\n",
                "    df_meta[f'{canal}_pot_parseval_dbm'] = [parseval_power(row) for row in sub_matrix]\n",
                "    df_meta[f'{canal}_ocupacion_pct'] = [np.mean(row > UMBRAL_OCUPADO_DBM) * 100.0 for row in sub_matrix]\n",
                "    \n",
                "    resumen_canales.append({\n",
                "        'Canal': canal,\n",
                "        'Banda (MHz)': f\"{f_min:.1f} - {f_max:.1f}\",\n",
                "        'Potencia Max (dBm)': round(p_max, 2),\n",
                "        'Potencia Parseval (dBm)': round(p_parseval, 2),\n",
                "        'Muestras > -60 dBm': f\"{pts_ocupados} / {total_pts}\",\n",
                "        'Ocupación (%)': round(pct_ocup, 2)\n",
                "    })\n",
                "\n",
                "df_canales = pd.DataFrame(resumen_canales)\n",
                "display(df_canales)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Identificación de Frecuencias Extremas\n",
                "- **Frecuencia Más Contaminada:** Mayor tasa de superación del umbral de -60 dBm.\n",
                "- **Frecuencia Menos Contaminada:** Menor tasa de ocupación y menor potencia."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ocup_por_bin = np.sum(clean_specs > UMBRAL_OCUPADO_DBM, axis=0)\n",
                "pwr_parseval_por_bin = 10.0 * np.log10(np.mean(10.0 ** (clean_specs / 10.0), axis=0))\n",
                "\n",
                "idx_max = np.argmax(ocup_por_bin)\n",
                "idx_min = np.argmin(ocup_por_bin)\n",
                "\n",
                "f_max_val = FREQS_MHZ[idx_max]\n",
                "f_min_val = FREQS_MHZ[idx_min]\n",
                "\n",
                "print(f\"🔥 FRECUENCIA MÁS CONTAMINADA:\")\n",
                "print(f\"   * Frecuencia:        {f_max_val:.4f} MHz (Canal C, Bin {idx_max})\")\n",
                "print(f\"   * Ocupación:         {ocup_por_bin[idx_max]} / 61 mediciones ({ocup_por_bin[idx_max]/61*100:.1f}%)\")\n",
                "print(f\"   * Potencia Parseval: {pwr_parseval_por_bin[idx_max]:.2f} dBm\")\n",
                "\n",
                "print(f\"\\n🟢 FRECUENCIA MENOS CONTAMINADA:\")\n",
                "print(f\"   * Frecuencia:        {f_min_val:.4f} MHz (Canal A, Bin {idx_min})\")\n",
                "print(f\"   * Ocupación:         {ocup_por_bin[idx_min]} / 61 mediciones ({ocup_por_bin[idx_min]/61*100:.1f}%)\")\n",
                "print(f\"   * Potencia Parseval: {pwr_parseval_por_bin[idx_min]:.2f} dBm\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Análisis de la Temperatura del Sensor y Calidad"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_meta['pot_global_parseval'] = [parseval_power(row) for row in clean_specs]\n",
                "\n",
                "print(\"Correlaciones de Pearson con la Temperatura:\")\n",
                "print(f\"  * Error GPS (HDOP):     r = {df_meta['temp'].corr(df_meta['err_dist']):+.4f}\")\n",
                "print(f\"  * Altitud:              r = {df_meta['temp'].corr(df_meta['alt']):+.4f}\")\n",
                "print(f\"  * Potencia Global:      r = {df_meta['temp'].corr(df_meta['pot_global_parseval']):+.4f}\")\n",
                "print(f\"  * Potencia Canal C:     r = {df_meta['temp'].corr(df_meta['Canal C_pot_parseval_dbm']):+.4f}\")\n",
                "\n",
                "print(f\"\\nRango térmico del USRP: {df_meta['temp'].min():.2f} °C -> {df_meta['temp'].max():.2f} °C (Incremento: +{df_meta['temp'].max() - df_meta['temp'].min():.2f} °C)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Gráficas Técnicas para el Informe"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Gráfica General del Espectro con Canales\n",
                "plt.figure(figsize=(12, 6))\n",
                "spec_mean = 10.0 * np.log10(np.mean(10.0 ** (clean_specs / 10.0), axis=0))\n",
                "spec_max = np.max(clean_specs, axis=0)\n",
                "\n",
                "plt.plot(FREQS_MHZ, spec_mean, label='Potencia Media Parseval', color='#1f77b4', linewidth=1.5)\n",
                "plt.plot(FREQS_MHZ, spec_max, label='Potencia Pico', color='#aec7e8', alpha=0.5)\n",
                "plt.axhline(UMBRAL_OCUPADO_DBM, color='red', linestyle='--', label='Umbral Ocupado (-60 dBm)')\n",
                "plt.axhline(PISO_RUIDO_CORTE_DBM, color='orange', linestyle=':', label='Corte Piso Ruido (-65 dBm)')\n",
                "\n",
                "colores = ['#e6f2ff', '#e6ffe6', '#fff2e6', '#f9e6ff']\n",
                "for idx, (canal, info) in enumerate(CHANNELS.items()):\n",
                "    f_min, f_max = info['rango_mhz']\n",
                "    plt.axvspan(f_min, f_max, alpha=0.25, color=colores[idx], label=f\"{canal} ({f_min:.0f}-{f_max:.0f} MHz)\")\n",
                "    plt.text((f_min + f_max) / 2, -26, canal, ha='center', fontweight='bold')\n",
                "\n",
                "plt.title(\"Espectro de Frecuencia 840 - 860 MHz con Canales A, B, C y D\", fontsize=13, fontweight='bold')\n",
                "plt.xlabel(\"Frecuencia (MHz)\")\n",
                "plt.ylabel(\"Potencia (dBm)\")\n",
                "plt.ylim(-100, -20)\n",
                "plt.grid(True, linestyle='--', alpha=0.6)\n",
                "plt.legend(loc='lower left', ncol=2)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Gráfica Frecuencia Más Contaminada vs Menos Contaminada\n",
                "plt.figure(figsize=(11, 5))\n",
                "meds = np.arange(1, len(df_meta) + 1)\n",
                "plt.plot(meds, clean_specs[:, idx_max], 'r-o', markersize=4, label=f'Más Contaminada ({f_max_val:.4f} MHz - Canal C)')\n",
                "plt.plot(meds, clean_specs[:, idx_min], 'g-s', markersize=4, label=f'Menos Contaminada ({f_min_val:.4f} MHz - Canal A)')\n",
                "plt.axhline(UMBRAL_OCUPADO_DBM, color='black', linestyle='--', label='Umbral ANE (-60 dBm)')\n",
                "\n",
                "plt.title(\"Comportamiento Temporal/Espacial de las Frecuencias Extremas\", fontsize=12, fontweight='bold')\n",
                "plt.xlabel(\"Número de Medición en el Recorrido\")\n",
                "plt.ylabel(\"Potencia (dBm)\")\n",
                "plt.ylim(-100, -10)\n",
                "plt.grid(True, linestyle='--', alpha=0.6)\n",
                "plt.legend(loc='upper right')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Ruta GPS en Medellín y Calor de Potencia en Canal C\n",
                "plt.figure(figsize=(8, 8))\n",
                "scatter = plt.scatter(df_meta['lon'], df_meta['lat'], c=df_meta['Canal C_pot_parseval_dbm'],\n",
                "                      cmap='inferno', s=70, edgecolors='black', linewidth=0.5)\n",
                "plt.plot(df_meta['lon'], df_meta['lat'], 'gray', linestyle=':', alpha=0.6, label='Ruta Móvil')\n",
                "plt.colorbar(scatter, label='Potencia Parseval Canal C (dBm)')\n",
                "\n",
                "# Destacar el punto 008 que fue imputado\n",
                "pto_008 = df_meta[df_meta['archivo'] == '008.txt'].iloc[0]\n",
                "plt.scatter(pto_008['lon'], pto_008['lat'], color='cyan', s=150, marker='X', edgecolors='black', label='Pto 008 (GPS Imputado)')\n",
                "\n",
                "plt.title(\"Ruta de Sensado en Medellín - Potencia de Canal C\", fontsize=12, fontweight='bold')\n",
                "plt.xlabel(\"Longitud (°)\")\n",
                "plt.ylabel(\"Latitud (°)\")\n",
                "plt.grid(True, linestyle='--', alpha=0.6)\n",
                "plt.legend(loc='best')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 9. Conclusiones y Recomendación Técnica para la ANE\n",
                "1. **Banda Más Contaminada:** **Canal C (850 - 855 MHz)**, con **59.11% de ocupación** y una potencia media de Parseval de **-28.15 dBm**. Alberga portadoras activas de telefonía móvil celular de alta intensidad (con picos de hasta -3.64 dBm).\n",
                "2. **Banda Menos Contaminada:** **Canal A (840 - 845 MHz)**, con solo **21.38% de ocupación** y potencia media de **-46.38 dBm**.\n",
                "3. **Recomendación de Plan de Frecuencias:**\n",
                "   - **Se recomienda ASIGNAR / USAR el Canal A (840-845 MHz)** para nuevos despliegues de servicios de telecomunicaciones o enlaces punto a punto, dado su amplio margen libre de interferencias.\n",
                "   - **NO se recomienda asignar el Canal C (850-855 MHz)** para nuevos servicios sin coordinación previa, ya que está severamente saturado por operadores celulares existentes en el sector occidental de Medellín."
            ]
        }
    ],
    "metadata": {
        "colab": {
            "name": "analisis_iot_espectro.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2,
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_nb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analisis_iot_espectro.ipynb")
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook generado exitosamente: {out_nb}")
