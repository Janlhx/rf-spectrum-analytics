#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de Monitoreo de Espectro RF (840 - 860 MHz)
Estudio para la Agencia Nacional del Espectro (ANE) - Medellín
Ejecutar: streamlit run dashboard_espectro.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Espectro RF | ANE Medellín",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inyección de estilos premium ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    /* Encabezado principal */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-left: 5px solid #e94560;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(233,69,96,0.25);
    }
    .hero-banner h1 { color: #ffffff; margin: 0; font-weight: 700; font-size: 1.9rem; }
    .hero-banner p  { color: #a0a8d0; margin: 6px 0 0; font-size: 0.95rem; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { color: #a0a8d0; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #e94560; font-size: 2rem; font-weight: 700; margin: 6px 0; }
    .kpi-sub   { color: #6b7194; font-size: 0.8rem; }

    /* Alerts */
    .alert-red   { background:#2d1b1b; border-left:4px solid #e94560; border-radius:8px; padding:12px 16px; color:#f0d0d0; margin:8px 0; }
    .alert-green { background:#1a2d1b; border-left:4px solid #56e945; border-radius:8px; padding:12px 16px; color:#d0f0d0; margin:8px 0; }
    .alert-info  { background:#1b1e2d; border-left:4px solid #457be9; border-radius:8px; padding:12px 16px; color:#d0d8f0; margin:8px 0; }

    .section-title {
        color:#ffffff; font-size:1.1rem; font-weight:700;
        border-bottom:2px solid #e94560; padding-bottom:6px; margin-bottom:16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f0c29; border-right: 1px solid rgba(255,255,255,0.06); }
    [data-testid="stSidebar"] * { color: #c5c9e8 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTES Y CARGA DE DATOS
# ==============================================================================
NUM_BINS = 1024
FREQS_MHZ = np.linspace(840.0, 860.0, NUM_BINS, endpoint=False)
UMBRAL_DBM = -60.0
PISO_CUT = -65.0
PISO_IMP = -95.0

CHANNELS = {
    'Canal A': {'rango': (840.0, 845.0), 'bins': (0, 256),  'color': '#4fc3f7'},
    'Canal B': {'rango': (845.0, 850.0), 'bins': (256, 512), 'color': '#81c784'},
    'Canal C': {'rango': (850.0, 855.0), 'bins': (512, 768), 'color': '#e94560'},
    'Canal D': {'rango': (855.0, 860.0), 'bins': (768, 1024),'color': '#ffb74d'},
}

@st.cache_data
def cargar_datos(data_dir: str):
    file_list = [f"{i:03d}.txt" for i in range(1, 62)]
    raw_specs, clean_specs, metadata = [], [], []

    for idx, fname in enumerate(file_list):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            parts = f.readline().strip().split(',')
        spec_raw = np.array([float(x) for x in parts[:1024]])
        spec_clean = np.where(spec_raw < PISO_CUT, PISO_IMP, spec_raw)
        raw_specs.append(spec_raw)
        clean_specs.append(spec_clean)
        metadata.append({
            'archivo': fname, 'id': idx + 1,
            'temp': float(parts[1024]), 'lon': float(parts[1025]),
            'lat': float(parts[1026]), 'alt': float(parts[1027]),
            'err_dist': float(parts[1028]),
        })

    df = pd.DataFrame(metadata)
    # Imputar GPS 008.txt
    bad = df[(df['lat'] == 0) | (df['lon'] == 0)].index
    for i in bad:
        df.loc[i, 'lat'] = (df.loc[i-1, 'lat'] + df.loc[i+1, 'lat']) / 2
        df.loc[i, 'lon'] = (df.loc[i-1, 'lon'] + df.loc[i+1, 'lon']) / 2
        df.loc[i, 'alt'] = (df.loc[i-1, 'alt'] + df.loc[i+1, 'alt']) / 2
        df.loc[i, 'gps_imputado'] = True
    df['gps_imputado'] = df['gps_imputado'].fillna(False)

    # Calcular indicadores por canal
    raw_arr = np.array(raw_specs)
    cln_arr = np.array(clean_specs)
    for ch, info in CHANNELS.items():
        b0, b1 = info['bins']
        sub = cln_arr[:, b0:b1]
        df[f'{ch}_parseval']  = [10*np.log10(np.mean(10**(r/10))) for r in sub]
        df[f'{ch}_ocupacion'] = [np.mean(r > UMBRAL_DBM)*100 for r in sub]

    # Potencia global
    df['parseval_global'] = [10*np.log10(np.mean(10**(r/10))) for r in cln_arr]

    return df, raw_arr, cln_arr

# ==============================================================================
# SIDEBAR - CONFIGURACIÓN
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, "medidas_2026_20")
    data_dir = st.text_input("📁 Ruta de datos:", value=default_data)

    if not os.path.exists(data_dir):
        st.error("Ruta no encontrada.")
        st.stop()

    canal_sel = st.selectbox("📡 Canal principal:", list(CHANNELS.keys()), index=2)
    umbral_usuario = st.slider("Umbral de ocupación (dBm):", -90, -40, -60, 5)

    st.markdown("---")
    st.markdown("**🎛️ Modo de Visualización del Espectro**")
    modo_espectro = st.radio(
        "Traza principal:",
        ["Promedio Parseval", "Max Hold", "Medición Individual"],
        index=0,
        help="Max Hold: valor máximo histórico por bin (envolvente superior de todas las mediciones)"
    )
    medicion_idx = 0
    if modo_espectro == "Medición Individual":
        medicion_idx = st.slider("Selecciona medición:", 1, 61, 1, 1,
                                  format="Medición %d")
    mostrar_maxhold   = st.checkbox("Superponer Max Hold", value=True)
    mostrar_parseval  = st.checkbox("Superponer Promedio Parseval", value=(modo_espectro != "Promedio Parseval"))

    st.markdown("---")
    st.markdown("**Parámetros del Sistema**")
    st.caption("• Frecuencia: 840 – 860 MHz")
    st.caption(f"• Bins espectrales: {NUM_BINS}")
    st.caption("• Δf por bin: 19.53 kHz")
    st.caption("• Mediciones: 61 puntos")
    st.markdown("---")
    st.caption("📡 Proyecto IoT – UPB 2026")

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
try:
    df, raw_arr, cln_arr = cargar_datos(data_dir)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ==============================================================================
# ENCABEZADO / HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <h1>📡 Dashboard de Ocupación de Espectro RF</h1>
    <p>Estudio de Monitoreo Móvil 840 – 860 MHz · Medellín, Colombia · Agencia Nacional del Espectro (ANE)</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECCIÓN 1: KPIs PRINCIPALES
# ==============================================================================
st.markdown('<div class="section-title">📊 Indicadores Globales de la Campaña</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

canal_c_sub = cln_arr[:, CHANNELS['Canal C']['bins'][0]:CHANNELS['Canal C']['bins'][1]]
canal_a_sub = cln_arr[:, CHANNELS['Canal A']['bins'][0]:CHANNELS['Canal A']['bins'][1]]

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Mediciones Totales</div>
        <div class="kpi-value">61</div>
        <div class="kpi-sub">Puntos de la ruta móvil</div>
    </div>""", unsafe_allow_html=True)

with k2:
    occ_c = np.mean(canal_c_sub > umbral_usuario) * 100
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Ocupación Canal C</div>
        <div class="kpi-value">{occ_c:.1f}%</div>
        <div class="kpi-sub">850 – 855 MHz (MÁS SATURADO)</div>
    </div>""", unsafe_allow_html=True)

with k3:
    occ_a = np.mean(canal_a_sub > umbral_usuario) * 100
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Ocupación Canal A</div>
        <div class="kpi-value">{occ_a:.1f}%</div>
        <div class="kpi-sub">840 – 845 MHz (MENOS SATURADO)</div>
    </div>""", unsafe_allow_html=True)

with k4:
    p_parseval_c = 10 * np.log10(np.mean(10**(canal_c_sub/10)))
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Potencia Parseval Canal C</div>
        <div class="kpi-value">{p_parseval_c:.1f} dBm</div>
        <div class="kpi-sub">Media ponderada lineal</div>
    </div>""", unsafe_allow_html=True)

with k5:
    delta_temp = df['temp'].max() - df['temp'].min()
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">ΔT Sensor USRP</div>
        <div class="kpi-value">+{delta_temp:.1f} °C</div>
        <div class="kpi-sub">{df['temp'].min():.1f} → {df['temp'].max():.1f} °C</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECCIÓN 2: ESPECTRO GLOBAL + MAPA
# ==============================================================================
col_spec, col_map = st.columns([6, 4], gap="medium")

with col_spec:
    st.markdown('<div class="section-title">📈 Espectro de Potencia por Canal (840 – 860 MHz) — Interactivo</div>', unsafe_allow_html=True)

    spec_mean_parseval = 10 * np.log10(np.mean(10**(cln_arr/10), axis=0))
    spec_maxhold       = np.max(cln_arr, axis=0)
    spec_individual    = cln_arr[medicion_idx - 1]

    fig_plotly = go.Figure()

    # ── Bandas sombreadas de los 4 canales ──────────────────────────────────
    colores_canal_fill = ['rgba(79,195,247,0.10)', 'rgba(129,199,132,0.10)',
                          'rgba(233,69,96,0.10)',  'rgba(255,183,77,0.10)']
    colores_canal_line = ['#4fc3f7', '#81c784', '#e94560', '#ffb74d']
    for idx, (canal, info) in enumerate(CHANNELS.items()):
        f0, f1 = info['rango']
        fig_plotly.add_vrect(
            x0=f0, x1=f1,
            fillcolor=colores_canal_fill[idx],
            line=dict(color=colores_canal_line[idx], width=1, dash='dot'),
            annotation_text=canal, annotation_position="top left",
            annotation_font=dict(color=colores_canal_line[idx], size=11)
        )

    # ── Max Hold ─────────────────────────────────────────────────────────────
    if mostrar_maxhold or modo_espectro == "Max Hold":
        fig_plotly.add_trace(go.Scatter(
            x=FREQS_MHZ, y=spec_maxhold,
            mode='lines',
            name='🔴 Max Hold (envolvente máxima)',
            line=dict(color='#ff6b6b', width=1.5, dash='dot'),
            hovertemplate='<b>Max Hold</b><br>Freq: %{x:.4f} MHz<br>Pot: %{y:.2f} dBm<extra></extra>'
        ))

    # ── Promedio Parseval ────────────────────────────────────────────────────
    if mostrar_parseval or modo_espectro == "Promedio Parseval":
        fig_plotly.add_trace(go.Scatter(
            x=FREQS_MHZ, y=spec_mean_parseval,
            mode='lines',
            name='🔵 Promedio Parseval',
            line=dict(color='#4fc3f7', width=2.5),
            fill='tozeroy', fillcolor='rgba(79,195,247,0.04)',
            hovertemplate='<b>Parseval</b><br>Freq: %{x:.4f} MHz<br>Pot: %{y:.2f} dBm<extra></extra>'
        ))

    # ── Medición Individual ──────────────────────────────────────────────────
    if modo_espectro == "Medición Individual":
        row = df.iloc[medicion_idx - 1]
        fig_plotly.add_trace(go.Scatter(
            x=FREQS_MHZ, y=spec_individual,
            mode='lines',
            name=f'📍 Medición {medicion_idx:03d} (lat={row["lat"]:.4f}°, lon={row["lon"]:.4f}°)',
            line=dict(color='#ffe066', width=2),
            hovertemplate='<b>Medición %{text}</b><br>Freq: %{x:.4f} MHz<br>Pot: %{y:.2f} dBm<extra></extra>',
            text=[f"{medicion_idx:03d}"]*NUM_BINS
        ))

    # ── Líneas de umbral ─────────────────────────────────────────────────────
    fig_plotly.add_hline(y=umbral_usuario, line=dict(color='#e94560', dash='dash', width=1.8),
                         annotation_text=f'Umbral ocupado ({umbral_usuario} dBm)',
                         annotation_font_color='#e94560')
    fig_plotly.add_hline(y=PISO_CUT, line=dict(color='#ffb74d', dash='dot', width=1.2),
                         annotation_text='Piso de ruido (-65 dBm)',
                         annotation_font_color='#ffb74d')

    # ── Layout oscuro ────────────────────────────────────────────────────────
    fig_plotly.update_layout(
        paper_bgcolor='#0f0c29',
        plot_bgcolor='#0f0c29',
        font=dict(color='#c5c9e8', family='Inter'),
        xaxis=dict(
            title='Frecuencia (MHz)', range=[840, 860],
            gridcolor='#2a2d45', zerolinecolor='#2a2d45',
            tickfont=dict(color='#a0a8d0'),
        ),
        yaxis=dict(
            title='Potencia (dBm)', range=[-100, -22],
            gridcolor='#2a2d45', zerolinecolor='#2a2d45',
            tickfont=dict(color='#a0a8d0')
        ),
        legend=dict(
            bgcolor='rgba(26,26,46,0.9)', bordercolor='#2a2d45',
            borderwidth=1, font=dict(size=11)
        ),
        hovermode='x unified',
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode='zoom',
        modebar_remove=['lasso2d', 'select2d'],
    )
    st.plotly_chart(fig_plotly, use_container_width=True, config={
        'displaylogo': False,
        'toImageButtonOptions': {'format': 'png', 'filename': 'espectro_rf', 'scale': 2}
    })

with col_map:
    st.markdown('<div class="section-title">🗺️ Ruta de Sensado en Medellín</div>', unsafe_allow_html=True)

    lat_c = df['lat'].mean()
    lon_c = df['lon'].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles='CartoDB dark_matter')

    # Línea de ruta
    ruta_coords = list(zip(df['lat'], df['lon']))
    folium.PolyLine(ruta_coords, color='#4fc3f7', weight=2, opacity=0.6).add_to(m)

    # Colorear puntos por canal seleccionado
    pot_col = f'{canal_sel}_parseval'
    pot_vals = df[pot_col].values
    vmin, vmax = pot_vals.min(), pot_vals.max()
    cmap = matplotlib.colormaps['inferno']

    for _, row in df.iterrows():
        norm = (row[pot_col] - vmin) / (vmax - vmin + 1e-9)
        rgba = cmap(norm)
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        popup_txt = (f"<b>{row['archivo']}</b><br>"
                     f"Lat: {row['lat']:.5f}<br>Lon: {row['lon']:.5f}<br>"
                     f"Parseval {canal_sel}: {row[pot_col]:.1f} dBm<br>"
                     f"Alt: {row['alt']:.0f} m | T: {row['temp']:.1f} °C")
        radius = 7 if row.get('gps_imputado', False) else 6
        border = '#00ffff' if row.get('gps_imputado', False) else color_hex
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=border, fill=True, fill_color=color_hex, fill_opacity=0.85,
            popup=folium.Popup(popup_txt, max_width=220)
        ).add_to(m)

    st_folium(m, height=380, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECCIÓN 3: TABLA COMPARATIVA DE CANALES
# ==============================================================================
st.markdown('<div class="section-title">📋 Tabla Comparativa de Canales – Resumen ANE</div>', unsafe_allow_html=True)

resumen_rows = []
for ch, info in CHANNELS.items():
    b0, b1 = info['bins']
    sub = cln_arr[:, b0:b1]
    pts_occ = np.sum(sub > umbral_usuario)
    total_pts = sub.size
    pct_occ = pts_occ / total_pts * 100
    p_parseval = 10 * np.log10(np.mean(10**(sub/10)))
    p_max = np.max(sub)
    verdict = "🔴 SATURADO" if pct_occ > 50 else ("🟡 MODERADO" if pct_occ > 25 else "🟢 LIBRE")
    resumen_rows.append({
        'Canal': ch,
        'Banda (MHz)': f"{info['rango'][0]:.1f} – {info['rango'][1]:.1f}",
        'Potencia Pico (dBm)': f"{p_max:.2f}",
        'Potencia Parseval (dBm)': f"{p_parseval:.2f}",
        f'Muestras > {umbral_usuario} dBm': f"{pts_occ:,} / {total_pts:,}",
        'Ocupación (%)': f"{pct_occ:.2f}%",
        'Estado': verdict
    })

df_resumen = pd.DataFrame(resumen_rows)
st.dataframe(df_resumen, use_container_width=True, hide_index=True)

# ==============================================================================
# SECCIÓN 4: EVOLUCIÓN TEMPORAL (SELECCIÓN POR CANAL)
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_evo, col_temp = st.columns(2, gap="medium")

with col_evo:
    st.markdown(f'<div class="section-title">📉 Evolución de Potencia Parseval – {canal_sel}</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    fig2.patch.set_facecolor('#0f0c29')
    ax2.set_facecolor('#0f0c29')
    color_ch = CHANNELS[canal_sel]['color']
    xs = np.arange(1, len(df)+1)
    ax2.plot(xs, df[f'{canal_sel}_parseval'], color=color_ch, linewidth=2, marker='o', markersize=4)
    ax2.fill_between(xs, df[f'{canal_sel}_parseval'], alpha=0.2, color=color_ch)
    ax2.axhline(umbral_usuario, color='#e94560', linestyle='--', linewidth=1.2, label=f'Umbral ({umbral_usuario} dBm)')
    ax2.set_xlabel("Medición #", color='#a0a8d0')
    ax2.set_ylabel("Potencia Parseval (dBm)", color='#a0a8d0')
    ax2.tick_params(colors='#a0a8d0')
    for s in ax2.spines.values():
        s.set_edgecolor('#2a2d45')
    ax2.grid(True, color='#2a2d45', linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#1a1a2e', edgecolor='#2a2d45', labelcolor='#c5c9e8', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with col_temp:
    st.markdown('<div class="section-title">🌡️ Temperatura del Sensor USRP vs Error GPS</div>', unsafe_allow_html=True)
    fig3, ax3a = plt.subplots(figsize=(7, 3.5))
    fig3.patch.set_facecolor('#0f0c29')
    ax3a.set_facecolor('#0f0c29')
    xs = np.arange(1, len(df)+1)
    ax3a.plot(xs, df['temp'], color='#e94560', linewidth=2, marker='o', markersize=3, label='Temperatura (°C)')
    ax3a.set_xlabel("Medición #", color='#a0a8d0')
    ax3a.set_ylabel("Temperatura (°C)", color='#e94560')
    ax3a.tick_params(colors='#a0a8d0')
    for s in ax3a.spines.values():
        s.set_edgecolor('#2a2d45')
    ax3a.grid(True, color='#2a2d45', linestyle='--', alpha=0.5)
    ax3b = ax3a.twinx()
    ax3b.set_facecolor('#0f0c29')
    ax3b.plot(xs, df['err_dist'], color='#4fc3f7', linewidth=1.8, linestyle='--', marker='^', markersize=3, label='Error GPS (m)')
    ax3b.set_ylabel("Error Distancia GPS (m)", color='#4fc3f7')
    ax3b.tick_params(colors='#4fc3f7')
    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labels1 + labels2, facecolor='#1a1a2e', edgecolor='#2a2d45', labelcolor='#c5c9e8', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ==============================================================================
# SECCIÓN 5: VEREDICTO TÉCNICO
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">⚖️ Juicio de Valor Técnico para la ANE</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="alert-red">
        <strong>🔴 Canal C (850–855 MHz)</strong><br>
        Ocupación: <b>59.11%</b> | Parseval: <b>-28.15 dBm</b><br>
        <em>SATURADO por operadores celulares. No se recomienda nuevas asignaciones sin coordinación previa.</em>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="alert-green">
        <strong>🟢 Canal A (840–845 MHz)</strong><br>
        Ocupación: <b>21.38%</b> | Parseval: <b>-46.38 dBm</b><br>
        <em>DISPONIBLE para nuevos servicios. Potencia de interferencia baja y ruido de fondo controlado.</em>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="alert-info">
        <strong>🔵 Canal B y D (845–860 MHz)</strong><br>
        Ocupación: <b>28.55%</b> y <b>25.44%</b><br>
        <em>MODERADOS. Viables con plan de coordinación de frecuencias entre operadores.</em>
    </div>""", unsafe_allow_html=True)

# ==============================================================================
# SECCIÓN 6: TABLA DE DATOS CRUDOS
# ==============================================================================
with st.expander("📂 Ver tabla de mediciones procesadas"):
    cols_show = ['archivo', 'id', 'lat', 'lon', 'alt', 'temp', 'err_dist',
                 'Canal A_parseval', 'Canal B_parseval', 'Canal C_parseval', 'Canal D_parseval',
                 'Canal A_ocupacion', 'Canal B_ocupacion', 'Canal C_ocupacion', 'Canal D_ocupacion']
    st.dataframe(df[cols_show].style.format({
        'lat': '{:.5f}', 'lon': '{:.5f}', 'alt': '{:.1f}',
        'temp': '{:.2f}', 'err_dist': '{:.1f}',
        'Canal A_parseval': '{:.2f}', 'Canal B_parseval': '{:.2f}',
        'Canal C_parseval': '{:.2f}', 'Canal D_parseval': '{:.2f}',
        'Canal A_ocupacion': '{:.1f}%', 'Canal B_ocupacion': '{:.1f}%',
        'Canal C_ocupacion': '{:.1f}%', 'Canal D_ocupacion': '{:.1f}%',
    }), use_container_width=True)

st.caption("📡 Dashboard de Espectro RF 840–860 MHz · Universidad Pontificia Bolivariana · IoT 2026-20")
