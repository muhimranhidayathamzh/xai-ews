# ═══════════════════════════════════════════════════════════
#  XAI-EWS v3.0 — Three-Level XAI Framework Demonstrator
#  Dashboard Streamlit | Luwu Raya, Sulawesi Selatan
#  Random seed: 42
# ═══════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-EWS v3.0 — Framework Demonstrator",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path data ────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
RES_DIR  = DATA_DIR / "results"

# ── Konstanta ────────────────────────────────────────────────
FITUR_LS = ["elevasi", "slope", "aspect", "curah_hujan", "ndvi", "jarak_sungai_km"]
FITUR_BJ = ["elevasi", "aspect", "twi", "curah_hujan", "ndvi", "jarak_sungai_km"]

LABEL_FITUR = {
    "elevasi"        : "Ketinggian Wilayah",
    "slope"          : "Kemiringan Lereng",
    "aspect"         : "Arah Lereng",
    "twi"            : "Indeks Topografi Basah",
    "curah_hujan"    : "Curah Hujan Tahunan",
    "ndvi"           : "Tutupan Vegetasi (NDVI)",
    "jarak_sungai_km": "Jarak ke Sungai",
}
UNIT_FITUR = {
    "elevasi"        : "m dpl",
    "slope"          : "°",
    "aspect"         : "°",
    "twi"            : "(indeks)",
    "curah_hujan"    : "mm/tahun",
    "ndvi"           : "(indeks 0–1)",
    "jarak_sungai_km": "km",
}

# Metrik framework — IP% dan LIME ρ dari Fase 3
IP_PCT  = 71.5   # Information Preservation %
IP_STD  = 15.0   # ± std
LIME_RHO = 0.724  # LIME-SHAP Spearman ρ

WARNA_RISK = {"high": "#d73027", "medium": "#fc8d59", "low": "#91cf60"}

# ── Load data (cached) ───────────────────────────────────────
@st.cache_data
def load_all():
    # Skor dan klasifikasi per kecamatan
    try:

        scores    = pd.read_csv(DATA_DIR / "kecamatan_scores.csv")

        shap_ls   = pd.read_csv(DATA_DIR / "kecamatan_shap_longsor.csv")

        shap_bj   = pd.read_csv(DATA_DIR / "kecamatan_shap_banjir.csv")

        l2        = pd.read_csv(DATA_DIR / "kecamatan_l2_profiles.csv")

        l3        = pd.read_csv(DATA_DIR / "kecamatan_l3_narratives.csv")

        crosshaz  = pd.read_csv(RES_DIR  / "crosshazard_table.csv")

        scorecard = pd.read_csv(RES_DIR  / "nb4_scorecard_comparison.csv")

        fidelity  = pd.read_csv(RES_DIR  / "fidelity_metrics.csv")

        coherence = pd.read_csv(RES_DIR  / "coherence_table.csv")

        gdf       = gpd.read_file(DATA_DIR / "luwu_raya.geojson")

        dibi_ls_raw = pd.read_csv(DATA_DIR / "bnpb_dibi_longsor_luwu.csv")

        dibi_bj_raw = pd.read_csv(DATA_DIR / "bnpb_dibi_banjir_luwu.csv")

    except Exception as e:

        st.error(f"❌ Error loading data: {e}")

        st.stop()

    # Aggregate DIBI per kabupaten
    dibi_ls_agg = (dibi_ls.groupby("Kabupaten")["Jumlah Kejadian"]
                   .sum().reset_index()
                   .rename(columns={"Jumlah Kejadian": "n_kejadian"}))
    dibi_bj_agg = (dibi_bj.groupby("Kabupaten")["Jumlah Kejadian"]
                   .sum().reset_index()
                   .rename(columns={"Jumlah Kejadian": "n_kejadian"}))

    # Centroid per kabupaten dari GeoJSON
    gdf_kab = (gdf.dissolve(by="NAME_2")
               .reset_index()[["NAME_2", "geometry"]])
    gdf_kab["lat"] = gdf_kab.geometry.centroid.y
    gdf_kab["lon"] = gdf_kab.geometry.centroid.x

    dibi_ls_agg = dibi_ls_agg.merge(
        gdf_kab[["NAME_2","lat","lon"]],
        left_on="Kabupaten", right_on="NAME_2", how="left"
    )
    dibi_bj_agg = dibi_bj_agg.merge(
        gdf_kab[["NAME_2","lat","lon"]],
        left_on="Kabupaten", right_on="NAME_2", how="left"
    )

    # Hitung metrik framework dari CSV
    r2_mean       = fidelity["r2_shap_vs_rf"].mean()
    coherence_mean = coherence["coherence_pct"].mean()

    return dict(
        scores=scores, shap_ls=shap_ls, shap_bj=shap_bj,
        l2=l2, l3=l3, crosshaz=crosshaz,
        scorecard=scorecard, fidelity=fidelity, coherence=coherence,
        gdf=gdf,
        dibi_ls=dibi_ls_agg, dibi_bj=dibi_bj_agg,
        r2_mean=r2_mean, coherence_mean=coherence_mean,
    )

data = load_all()
st.sidebar.success('✅ Data loaded')

# ── Navigasi sidebar ─────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/"
    "Flag_of_Indonesia.svg/320px-Flag_of_Indonesia.svg.png",
    width=80,
)
st.sidebar.title("XAI-EWS v3.0")
st.sidebar.caption("Three-Level XAI Framework\nDemonstrasi: Luwu Raya, Sulsel")
st.sidebar.divider()

halaman = st.sidebar.radio(
    "Navigasi",
    ["🔬 Framework XAI", "🗺️ Peta Regional", "📊 Teknis & Export"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "**Model:** Random Forest Calibrated  \n"
    "**Hazard:** Longsor + Banjir  \n"
    "**Wilayah:** 44 Kecamatan Luwu Raya  \n"
    "**Framework:** Three-Level XAI"
)
