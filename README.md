# RF Spectrum Analytics (840 - 860 MHz)

> End-to-end ETL pipeline, spatial signal analysis, interference source estimation, and interactive Streamlit dashboard for RF spectrum monitoring in Medellin, Colombia (ANE standard).

## Overview

This project processes mobile RF spectrum measurements (`001.txt` - `061.txt`) across the **840 - 860 MHz** band, segmented into four 5 MHz sub-bands (Channels A, B, C, and D; 256 frequency bins each, 1024 total bins per sweep) paired with GPS telemetry.

### Key Components

1. **ETL & Data Imputation (`etl_analisis_espectro.py`)**
   - Parses sequential raw spectrum sweeps and GPS coordinates.
   - Applies noise-floor thresholding (readings below `-65.0 dBm` standardized to `-95.0 dBm` per ANE / domain rules).
   - Repairs GPS signal loss (`008.txt` with `(0, 0)` fix) via linear interpolation using adjacent sweeps (`007.txt` and `009.txt`).
   - Outputs structured datasets into `salida_procesada/`.

2. **Spectral & Quality Correlation Analysis (`analisis_espectro.py`, `correlacion_calidad.py`)**
   - Computes per-channel power statistics and occupancy indicators across Channels A–D.
   - Evaluates correlation between signal quality metrics and spatial/frequency behavior.

3. **Geographic Source Estimation (`estimacion_fuentes.py`)**
   - Estimates likely RF interference/emission source coordinates using power-weighted spatial aggregation over the top 25% highest-power measurements (`p_lin ** alpha`).

4. **Interactive Dashboard (`dashboard_espectro.py`) & Notebook (`analisis_iot_espectro.ipynb`)**
   - Built with **Streamlit**, **Folium**, **Pandas**, **NumPy**, and **Matplotlib** for geospatial visualization and spectral inspection.

## Tech Stack

- Python 3.10+
- Pandas, NumPy, Matplotlib
- Streamlit, Folium, `streamlit-folium`

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ETL pipeline
python etl_analisis_espectro.py

# 3. Run spatial source estimation
python estimacion_fuentes.py

# 4. Launch the interactive dashboard
streamlit run dashboard_espectro.py
```

## Project Structure

```text
.
├── etl_analisis_espectro.py   # ETL, cleaning, GPS interpolation, channel segmentation
├── analisis_espectro.py       # Channel-level spectral analysis (Channels A-D)
├── correlacion_calidad.py     # Signal quality correlation analysis
├── estimacion_fuentes.py      # Power-weighted geographic source estimation
├── dashboard_espectro.py      # Streamlit + Folium interactive dashboard
├── explorar_datos.py          # Initial dataset inspection
├── secuencia_medidas.py       # Sequential sweep ordering check
├── generar_notebook.py        # Script to generate the analysis notebook
├── analisis_iot_espectro.ipynb
├── medidas_2026_20/           # Raw mobile RF measurements
├── salida_procesada/          # Processed CSVs and outputs
└── requirements.txt
```
