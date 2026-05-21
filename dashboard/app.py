import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(
    page_title="XAI-EWS v3.0",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
RES_DIR  = DATA_DIR / "results"

FITUR_LS = ["elevasi","slope","aspect","curah_hujan","ndvi","jarak_sungai_km"]
FITUR_BJ = ["elevasi","aspect","twi","curah_hujan","ndvi","jarak_sungai_km"]

LABEL_FITUR = {
    "elevasi":"Ketinggian Wilayah","slope":"Kemiringan Lereng",
    "aspect":"Arah Lereng","twi":"Indeks Topografi Basah",
    "curah_hujan":"Curah Hujan Tahunan","ndvi":"Tutupan Vegetasi (NDVI)",
    "jarak_sungai_km":"Jarak ke Sungai",
}
UNIT_FITUR = {
    "elevasi":"m dpl","slope":"derajat","aspect":"derajat",
    "twi":"(indeks)","curah_hujan":"mm/tahun",
    "ndvi":"(indeks 0-1)","jarak_sungai_km":"km",
}

EV_LONGSOR = 0.4975
EV_BANJIR  = 0.4664
IP_PCT     = 71.5
IP_STD     = 15.0
LIME_RHO   = 0.724
WARNA_RISK = {"high":"#d73027","medium":"#fc8d59","low":"#91cf60"}
LABEL_RISK = {"high":"Risiko Tinggi","medium":"Risiko Sedang","low":"Risiko Rendah"}

def fmt_val(fitur, val):
    if fitur in ("elevasi","slope","aspect"):
        return f"{val:.1f} {UNIT_FITUR[fitur]}"
    elif fitur == "curah_hujan":
        return f"{val:.0f} {UNIT_FITUR[fitur]}"
    elif fitur == "jarak_sungai_km":
        return f"{val:.2f} {UNIT_FITUR[fitur]}"
    else:
        return f"{val:.3f} {UNIT_FITUR[fitur]}"

@st.cache_data
def load_all():
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
        st.error(f"Error loading data: {e}")
        st.stop()

    dibi_ls_agg = (dibi_ls_raw.groupby("Kabupaten")["Jumlah Kejadian"]
                   .sum().reset_index()
                   .rename(columns={"Jumlah Kejadian":"n_kejadian"}))
    dibi_bj_agg = (dibi_bj_raw.groupby("Kabupaten")["Jumlah Kejadian"]
                   .sum().reset_index()
                   .rename(columns={"Jumlah Kejadian":"n_kejadian"}))

    gdf_kab = gdf.dissolve(by="NAME_2").reset_index()[["NAME_2","geometry"]]
    gdf_kab["lat"] = gdf_kab.geometry.centroid.y
    gdf_kab["lon"] = gdf_kab.geometry.centroid.x

    dibi_ls_agg = dibi_ls_agg.merge(
        gdf_kab[["NAME_2","lat","lon"]], left_on="Kabupaten", right_on="NAME_2", how="left")
    dibi_bj_agg = dibi_bj_agg.merge(
        gdf_kab[["NAME_2","lat","lon"]], left_on="Kabupaten", right_on="NAME_2", how="left")

    return dict(
        scores=scores, shap_ls=shap_ls, shap_bj=shap_bj,
        l2=l2, l3=l3, crosshaz=crosshaz,
        scorecard=scorecard, fidelity=fidelity, coherence=coherence,
        gdf=gdf, dibi_ls=dibi_ls_agg, dibi_bj=dibi_bj_agg,
        r2_mean=fidelity["r2_shap_vs_rf"].mean(),
        coherence_mean=coherence["coherence_pct"].mean(),
    )

data = load_all()

st.sidebar.title("XAI-EWS v3.0")
st.sidebar.caption("Three-Level XAI Framework\nLuwu Raya, Sulawesi Selatan")
st.sidebar.divider()
halaman = st.sidebar.radio(
    "Navigasi",
    ["🔬 Framework XAI","🗺️ Peta Regional","📊 Teknis & Export"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    "**Model:** Random Forest Calibrated\n"
    "**Hazard:** Longsor + Banjir\n"
    "**Wilayah:** 44 Kecamatan Luwu Raya\n"
    "**Framework:** Three-Level XAI"
)

# ── HALAMAN 1 ────────────────────────────────────────────────
if halaman == "🔬 Framework XAI":
    st.title("🔬 Three-Level XAI Framework")
    st.caption("Demonstrasi framework explainability tiga level · Luwu Raya, Sulawesi Selatan")
    st.divider()

    col_sel, col_haz, col_metric = st.columns([3,1,1])
    with col_sel:
        kec_pilihan = st.selectbox(
            "📍 Pilih Kecamatan",
            sorted(data["scores"]["NAME_3"].tolist()),
            help="Pilih kecamatan untuk melihat journey L1 → L2 → L3",
        )
    with col_haz:
        hazard_pilihan = st.radio("Hazard", ["Longsor","Banjir"], horizontal=False)

    hazard_key = hazard_pilihan.lower()
    fitur_list = FITUR_LS if hazard_key == "longsor" else FITUR_BJ
    shap_df    = data["shap_ls"] if hazard_key == "longsor" else data["shap_bj"]
    base_ev    = EV_LONGSOR if hazard_key == "longsor" else EV_BANJIR

    row     = data["scores"][data["scores"]["NAME_3"] == kec_pilihan].iloc[0]
    prob    = float(row[f"prob_{hazard_key}"])
    risk    = str(row[f"risk_{hazard_key}"])
    kab     = row["NAME_2"]
    no_data = bool(row.get("no_data_longsor", False)) if hazard_key == "longsor" else False

    with col_metric:
        st.metric(f"Probabilitas {hazard_pilihan}", f"{prob*100:.1f}%")
        st.markdown(
            f'<span style="background:{WARNA_RISK.get(risk,"#999")};color:white;' +
            f'padding:3px 12px;border-radius:12px;font-size:0.85em;font-weight:600">' +
            f'{LABEL_RISK.get(risk,"—")}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(f"#### {kec_pilihan} · Kab. {kab}")

    if no_data:
        st.warning(
            "Tidak ada titik longsor teridentifikasi di kecamatan ini. "
            "Probabilitas di-impute sebagai 0 — analisis L2 dan L3 tidak tersedia."
        )
        st.stop()

    st.divider()

    # L1 Global
    st.subheader("🌐 Level 1 — Analisis Global")
    st.caption("Kepentingan setiap faktor lingkungan secara keseluruhan — perbandingan longsor vs banjir.")

    df_cross = data["crosshaz"].copy()
    df_cross["label"] = df_cross["fitur"].map(LABEL_FITUR).fillna(df_cross["fitur"])

    col_l1a, col_l1b = st.columns([2,1])
    with col_l1a:
        fig_l1 = go.Figure()
        fig_l1.add_trace(go.Bar(
            name="Longsor", y=df_cross["label"], x=df_cross["norm_shap_longsor"],
            orientation="h", marker_color="#c0392b", opacity=0.85,
        ))
        fig_l1.add_trace(go.Bar(
            name="Banjir", y=df_cross["label"], x=df_cross["norm_shap_banjir"],
            orientation="h", marker_color="#2980b9", opacity=0.85,
        ))
        fig_l1.update_layout(
            barmode="group", title="Kepentingan Fitur Ternormalisasi (0-1)",
            xaxis_title="Nilai SHAP Ternormalisasi", height=320,
            margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_l1, use_container_width=True)
    with col_l1b:
        st.markdown("**Ranking Lintas Hazard**")
        df_rank = df_cross[["label","rank_longsor","rank_banjir","kategori"]].copy()
        df_rank.columns = ["Fitur","Rank Longsor","Rank Banjir","Kategori"]
        st.dataframe(df_rank.sort_values("Rank Longsor"),
                     use_container_width=True, hide_index=True, height=280)
        st.caption("consistent = penting di kedua hazard")
    st.divider()

    # L2 Lokal
    st.subheader("🎯 Level 2 — Analisis Lokal")
    st.caption("Kontribusi spesifik setiap faktor untuk kecamatan terpilih.")

    shap_row = shap_df[shap_df["NAME_3"] == kec_pilihan]
    l2_row   = data["l2"][
        (data["l2"]["NAME_3"] == kec_pilihan) & (data["l2"]["hazard"] == hazard_key)]

    col_wf, col_card = st.columns([3,2])
    with col_wf:
        if shap_row.empty:
            st.info("Data SHAP tidak tersedia untuk kecamatan ini.")
        else:
            sr        = shap_row.iloc[0]
            shap_dict = {f: float(sr[f"shap_{f}"]) for f in fitur_list}
            val_dict  = {f: float(sr[f"val_{f}"])  for f in fitur_list}
            sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]))
            y_labels = [f"{LABEL_FITUR.get(f,f)} ({fmt_val(f,val_dict[f])})"
                        for f,_ in sorted_items]
            y_full   = ["Baseline (rata-rata model)"] + y_labels + ["Prediksi Akhir"]
            x_vals   = [base_ev] + [v for _,v in sorted_items] + [0]
            measures = ["absolute"] + ["relative"]*len(sorted_items) + ["total"]
            fig_wf = go.Figure(go.Waterfall(
                orientation="h", measure=measures, y=y_full, x=x_vals,
                connector={"line":{"color":"#cccccc","width":0.5,"dash":"dot"}},
                increasing={"marker":{"color":"#d73027"}},
                decreasing={"marker":{"color":"#91cf60"}},
                totals={"marker":{"color":"#4575b4"}},
                textposition="outside",
                text=[f"{v:+.3f}" if 0<i<len(x_vals)-1 else f"{v:.3f}"
                      for i,v in enumerate(x_vals)],
            ))
            fig_wf.update_layout(
                title="SHAP Waterfall — Kontribusi Per Fitur",
                xaxis_title="Probabilitas", xaxis=dict(range=[0,1.15]),
                height=360, margin=dict(l=10,r=40,t=40,b=10),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    with col_card:
        st.markdown("**Profil Risiko Kecamatan**")
        if not l2_row.empty:
            lr       = l2_row.iloc[0]
            mean_reg = data["scores"][f"prob_{hazard_key}"].mean()
            delta    = (prob - mean_reg) * 100
            st.metric("Probabilitas", f"{prob*100:.1f}%",
                      delta=f"{delta:+.1f}% vs rata-rata ({mean_reg*100:.1f}%)",
                      delta_color="inverse")
            st.markdown("**Faktor Pendorong:**")
            for i, c in enumerate(["pendorong_1","pendorong_2","pendorong_3"], 1):
                f = lr.get(c)
                if pd.notna(f) and not shap_row.empty:
                    sv = float(shap_row.iloc[0].get(f"shap_{f}", 0))
                    vv = float(shap_row.iloc[0].get(f"val_{f}", 0))
                    st.markdown(f"{i}. **{LABEL_FITUR.get(f,f)}** — {fmt_val(f,vv)} *(SHAP: +{sv:.3f})*")
            st.markdown("**Faktor Pelindung:**")
            for i, c in enumerate(["pelindung_1","pelindung_2"], 1):
                f = lr.get(c)
                if pd.notna(f) and not shap_row.empty:
                    sv = float(shap_row.iloc[0].get(f"shap_{f}", 0))
                    vv = float(shap_row.iloc[0].get(f"val_{f}", 0))
                    st.markdown(f"{i}. **{LABEL_FITUR.get(f,f)}** — {fmt_val(f,vv)} *(SHAP: {sv:.3f})*")
        else:
            st.info("Profil L2 tidak tersedia.")
    st.divider()

    # L3 Narasi
    st.subheader("📝 Level 3 — Narasi & Rekomendasi")
    st.caption("Penjelasan dalam Bahasa Indonesia untuk non-technical stakeholders.")

    l3_row = data["l3"][
        (data["l3"]["NAME_3"] == kec_pilihan) & (data["l3"]["hazard"] == hazard_key)]

    if not l3_row.empty:
        narasi  = l3_row.iloc[0]["narasi"]
        kalimat = [k.strip() for k in narasi.split(".") if len(k.strip()) > 10]
        icons   = ["⚠️","🔍","✅"]
        headers = ["Status Risiko","Faktor & Kondisi","Rekomendasi DRR"]
        warna_l = WARNA_RISK.get(risk,"#999")
        cols_l3 = st.columns(3)
        for col_l3, icon, header, kal in zip(cols_l3, icons, headers, kalimat[:3]):
            with col_l3:
                st.markdown(
                    f'<div style="border-left:4px solid {warna_l};border-radius:4px;' +
                    f'padding:14px;min-height:150px">' +
                    f'<strong>{icon} {header}</strong><br><br>' +
                    f'<span style="font-size:0.9em">{kal}.</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Narasi L3 tidak tersedia.")
    st.divider()

    # Framework Validation Panel
    st.subheader("🔬 Validasi Framework")
    st.caption("Bukti empiris bahwa Three-Level XAI Framework ini terukur dan konsisten.")
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.metric("Fidelitas SHAP (R²)", f"{data['r2_mean']:.3f}")
        st.caption("Penjelasan SHAP faithful terhadap prediksi model.")
    with vc2:
        st.metric("Koherensi L2→L3", f"{data['coherence_mean']:.1f}%")
        st.caption("Narasi Level 3 konsisten dengan SHAP Level 2.")
    with vc3:
        st.metric("Preservasi Informasi", f"{IP_PCT:.0f}% ± {IP_STD:.0f}%")
        st.caption("Magnitude SHAP tercakup dalam narasi Level 3.")
    with vc4:
        st.metric("Konsistensi LIME-SHAP", f"rho={LIME_RHO:.3f}")
        st.caption("Cross-validation antar metode XAI.")

# ── HALAMAN 2 ────────────────────────────────────────────────
elif halaman == "🗺️ Peta Regional":
    st.title("🗺️ Peta Kerentanan Regional")
    st.caption("Distribusi spasial probabilitas kerentanan — 44 Kecamatan Luwu Raya")
    st.divider()

    col_p1, col_p2 = st.columns([1,1])
    with col_p1:
        hazard_peta = st.radio("Tampilkan hazard:", ["Longsor","Banjir"], horizontal=True)
    with col_p2:
        show_dibi = st.checkbox("Tampilkan overlay kejadian BNPB DIBI", value=False)

    hazard_peta_key = hazard_peta.lower()
    dibi_data = data["dibi_ls"] if hazard_peta_key == "longsor" else data["dibi_bj"]

    gdf_map = data["gdf"].merge(
        data["scores"][["NAME_3",f"prob_{hazard_peta_key}",f"risk_{hazard_peta_key}"]],
        on="NAME_3", how="left",
    )

    m = folium.Map(location=[-2.8,120.5], zoom_start=8, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf_map.to_json(),
        data=data["scores"],
        columns=["NAME_3",f"prob_{hazard_peta_key}"],
        key_on="feature.properties.NAME_3",
        fill_color="YlOrRd", fill_opacity=0.75,
        line_opacity=0.4, line_color="white",
        legend_name=f"Probabilitas {hazard_peta} (0-1)",
        nan_fill_color="#cccccc", nan_fill_opacity=0.4,
    ).add_to(m)

    gdf_tt = gdf_map.copy()
    gdf_tt[f"prob_{hazard_peta_key}"] = gdf_tt[f"prob_{hazard_peta_key}"].fillna(0).round(3)
    folium.GeoJson(
        json.loads(gdf_tt.to_json()),
        style_function=lambda x: {"fillOpacity":0,"weight":0},
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME_3","NAME_2",f"prob_{hazard_peta_key}",f"risk_{hazard_peta_key}"],
            aliases=["Kecamatan:","Kabupaten:","Probabilitas:","Kelas Risiko:"],
        ),
    ).add_to(m)

    if show_dibi:
        for _, drow in dibi_data.iterrows():
            if pd.notna(drow.get("lat")):
                n = int(drow["n_kejadian"])
                folium.CircleMarker(
                    location=[drow["lat"], drow["lon"]],
                    radius=min(n*2+5, 30),
                    color="#333333", fill=True, fill_color="#555555",
                    fill_opacity=0.25, weight=1.5,
                    tooltip=f"{drow['Kabupaten']}: {n} kejadian {hazard_peta.lower()} tercatat (BNPB DIBI)",
                ).add_to(m)

    st_folium(m, use_container_width=True, height=520, returned_objects=[])

    st.markdown(
        '<div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">' +
        '<span style="background:#d73027;color:white;padding:2px 10px;border-radius:8px">■ Risiko Tinggi</span>' +
        '<span style="background:#fc8d59;color:white;padding:2px 10px;border-radius:8px">■ Risiko Sedang</span>' +
        '<span style="background:#91cf60;color:white;padding:2px 10px;border-radius:8px">■ Risiko Rendah</span>' +
        '<span style="background:#cccccc;color:#333;padding:2px 10px;border-radius:8px">■ Tidak Ada Data</span>' +
        '</div>',
        unsafe_allow_html=True,
    )
    if show_dibi:
        st.caption("Lingkaran abu-abu = kejadian historis per kabupaten (BNPB DIBI).")

# ── HALAMAN 3 ────────────────────────────────────────────────
elif halaman == "📊 Teknis & Export":
    st.title("📊 Detail Teknis & Export")
    st.caption("Bukti teknis model dan framework · Download data prediksi")
    st.divider()

    st.subheader("🏆 Model Selection Scorecard")
    st.caption("Lima model dibandingkan dengan multi-criteria scorecard.")
    sc = data["scorecard"].copy()
    num_cols = sc.select_dtypes("number").columns.tolist()
    try:
        st.dataframe(sc.style.highlight_max(subset=num_cols, color="#d4edda", axis=0),
                     use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(sc, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("🔬 Metrik Evaluasi Framework")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**Fidelity SHAP per Hazard**")
        fi = data["fidelity"].rename(columns={
            "hazard":"Hazard","r2_shap_vs_rf":"R² SHAP",
            "mae":"MAE","max_error":"Max Error",
            "n_test":"N Test","interpretasi":"Interpretasi"})
        fi["Hazard"] = fi["Hazard"].str.title()
        st.dataframe(fi, use_container_width=True, hide_index=True)
    with col_f2:
        st.markdown("**Koherensi L2→L3 per Kategori**")
        co = data["coherence"][["hazard","kategori","k","overlap","coherence_pct"]].copy()
        co.columns = ["Hazard","Kategori","K Fitur","Overlap","Koherensi (%)"]
        co["Hazard"] = co["Hazard"].str.title()
        co["Kategori"] = co["Kategori"].str.title()
        st.dataframe(co, use_container_width=True, hide_index=True)

    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("SHAP Fidelity R²", f"{data['r2_mean']:.3f}")
    m2.metric("Koherensi L2→L3", f"{data['coherence_mean']:.1f}%")
    m3.metric("Preservasi Informasi", f"{IP_PCT}% +/- {IP_STD}%")
    m4.metric("LIME-SHAP rho", f"{LIME_RHO:.3f}")
    st.divider()

    st.subheader("📥 Download Data")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "📥 Prediksi Semua Kecamatan (CSV)",
            data=data["scores"].to_csv(index=False).encode("utf-8"),
            file_name="xai_ews_v3_prediksi_kecamatan.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption("Probabilitas dan kelas risiko longsor + banjir per kecamatan.")
    with col_d2:
        st.download_button(
            "📥 Narasi Level 3 (CSV)",
            data=data["l3"].to_csv(index=False).encode("utf-8"),
            file_name="xai_ews_v3_narasi_l3.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption("Narasi 3 kalimat Bahasa Indonesia per kecamatan per hazard.")
