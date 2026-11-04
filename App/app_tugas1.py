import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR.parent / "Tugas_1" / "artifacts"

st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Mengikuti warna tema aktif Streamlit (Light / Dark Mode) */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    /* Header sengaja tetap biru pada kedua tema */
    .hero {
        padding: 1.55rem 1.8rem;
        border-radius: 18px;
        color: #ffffff;
        background: linear-gradient(120deg, #102a43 0%, #1769aa 58%, #2b8ac6 100%);
        box-shadow: 0 10px 28px rgba(16, 42, 67, .14);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 1.85rem;
        margin: 0;
    }
    .hero p {
        color: #e5f2ff !important;
        margin: .45rem 0 0 0;
        font-size: .98rem;
    }

    /* Teks pendukung mengikuti warna tema aktif */
    .section-note {
        color: var(--text-color);
        opacity: .72;
        font-size: .9rem;
        margin-top: -.4rem;
    }

    /* Container form dan panel mengikuti secondary background */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px;
        border: 1px solid var(--border-color, rgba(128, 128, 128, .25));
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }

    /* Metric card mengikuti tema aktif, tidak dipaksa putih */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, .25));
        border-radius: 13px;
        padding: 14px 16px;
    }

    /* Elemen teks utama mengikuti warna teks tema */
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] small,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: var(--text-color);
    }

    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 650;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    required_files = [
        "bank_marketing_mlp.keras",
        "scaler.pkl",
        "feature_columns.pkl",
        "preprocessing_metadata.pkl",
        "model_info.json",
    ]
    missing = [name for name in required_files if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "File artefak belum lengkap: " + ", ".join(missing)
        )

    model = tf.keras.models.load_model(
        ARTIFACT_DIR / "bank_marketing_mlp.keras"
    )
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.pkl")
    metadata = joblib.load(ARTIFACT_DIR / "preprocessing_metadata.pkl")

    with open(ARTIFACT_DIR / "model_info.json", "r", encoding="utf-8") as file:
        model_info = json.load(file)

    return model, scaler, feature_columns, metadata, model_info


try:
    model, scaler, feature_columns, metadata, model_info = load_artifacts()
except Exception as error:
    st.error("Model atau artefak preprocessing tidak dapat dimuat.")
    st.code(str(error))
    st.info(
        "Jalankan notebook sampai bagian penyimpanan artefak. "
        "Pastikan folder artifacts berada satu direktori dengan app.py."
    )
    st.stop()

def prepare_input(raw_input):
    """
    Mengikuti preprocessing notebook:
    - merekonstruksi fitur engineering,
    - menyelaraskan kolom asli,
    - imputasi memakai median/modus data train,
    - one-hot encoding drop_first=True,
    - menyelaraskan feature_columns,
    - transform menggunakan scaler hasil training.
    """
    frame = pd.DataFrame([raw_input])

    if "pdays_contacted" in metadata["raw_input_columns"]:
        frame["pdays_contacted"] = (
            pd.to_numeric(frame["pdays"], errors="coerce") != -1
        ).astype("int8")

    if "balance_per_age" in metadata["raw_input_columns"]:
        age_values = pd.to_numeric(frame["age"], errors="coerce").replace(0, np.nan)
        balance_values = pd.to_numeric(frame["balance"], errors="coerce")
        frame["balance_per_age"] = (
            balance_values / age_values
        ).replace([np.inf, -np.inf], np.nan)

    frame = frame.reindex(columns=metadata["raw_input_columns"])

    for col in metadata["numeric_features"]:
        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).fillna(metadata["medians"][col])

    for col in metadata["categorical_features"]:
        frame[col] = frame[col].fillna(
            metadata["modes"][col]
        ).astype(str)

    encoded = pd.get_dummies(frame, drop_first=True, dtype=float)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(encoded).astype("float32")
    return scaled

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Bank Deposit Predictor</h1>
        <p>Prediksi potensi nasabah berlangganan deposito berjangka
        menggunakan Deep Learning Multi-Layer Perceptron (MLP).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

official_threshold = float(
    model_info.get("threshold", metadata.get("threshold", 0.62))
)

with st.sidebar:
    st.markdown("## 🏦 Bank Analytics")
    st.caption("MLP · Bank Marketing Dataset")
    st.divider()

    st.markdown("### Informasi Model")
    st.write(f"**Model terpilih:** {model_info.get('selected_model', metadata.get('selected_model', 'MLP'))}")
    st.write(f"**Jumlah fitur:** {len(feature_columns)}")
    st.write(f"**Threshold hasil validation:** {official_threshold:.2f}")
    st.success("Model dan artefak berhasil dimuat")

    st.divider()
    st.markdown("### Pengaturan Keputusan")
    use_custom_threshold = st.toggle(
        "Atur threshold secara manual",
        value=False,
        help="Jika dimatikan, aplikasi memakai threshold final hasil validation."
    )

    if use_custom_threshold:
        threshold = st.slider(
            "Threshold prediksi",
            min_value=0.10,
            max_value=0.90,
            value=official_threshold,
            step=0.01,
        )
        st.caption(
            "Threshold manual hanya mengubah keputusan YES/NO, "
            "bukan melatih ulang model."
        )
    else:
        threshold = official_threshold
        st.info(f"Threshold final digunakan: {threshold:.2f}")

st.markdown("## Data Nasabah")
st.markdown(
    '<p class="section-note">Lengkapi profil dan riwayat kampanye untuk memperoleh estimasi prediksi.</p>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    profile_col, campaign_col = st.columns(2, gap="large")

    with profile_col:
        with st.container(border=True):
            st.markdown("### 👤 Profil Nasabah")
            age = st.number_input(
                "Usia nasabah (tahun)", min_value=18, max_value=100, value=35
            )
            job = st.selectbox(
                "Pekerjaan",
                [
                    "admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown",
                ],
                index=4,
            )
            marital = st.selectbox(
                "Status pernikahan",
                ["divorced", "married", "single"],
                index=1,
            )
            education = st.selectbox(
                "Pendidikan",
                ["primary", "secondary", "tertiary", "unknown"],
                index=2,
            )
            default = st.radio(
                "Memiliki kredit macet?",
                ["no", "yes"],
                horizontal=True,
            )
            balance = st.number_input(
                "Saldo rata-rata tahunan (€)",
                min_value=-100000,
                max_value=1000000,
                value=1500,
                step=100,
            )
            housing = st.radio(
                "Memiliki pinjaman perumahan?",
                ["no", "yes"],
                horizontal=True,
            )
            loan = st.radio(
                "Memiliki pinjaman pribadi?",
                ["no", "yes"],
                horizontal=True,
            )

    with campaign_col:
        with st.container(border=True):
            st.markdown("### 📞 Riwayat Kontak Kampanye")
            contact = st.selectbox(
                "Jenis komunikasi",
                ["cellular", "telephone", "unknown"],
            )
            day_of_week = st.number_input(
                "Hari/tanggal kontak terakhir (day_of_week)",
                min_value=1,
                max_value=31,
                value=15,
                help="Gunakan nilai numerik sesuai kolom day_of_week pada dataset UCI."
            )
            month = st.selectbox(
                "Bulan kontak terakhir",
                [
                    "jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec",
                ],
                index=4,
            )
            campaign = st.number_input(
                "Jumlah kontak pada kampanye ini",
                min_value=1,
                max_value=100,
                value=2,
            )
            pdays = st.number_input(
                "Hari sejak kontak kampanye sebelumnya",
                min_value=-1,
                max_value=999,
                value=-1,
                help="-1 berarti nasabah belum pernah dihubungi pada kampanye sebelumnya."
            )
            previous = st.number_input(
                "Jumlah kontak sebelum kampanye ini",
                min_value=0,
                max_value=100,
                value=0,
            )
            poutcome = st.selectbox(
                "Hasil kampanye sebelumnya",
                ["failure", "other", "success", "unknown"],
                index=3,
            )

    submitted = st.form_submit_button(
        "🔮 Prediksi Potensi Nasabah",
        type="primary",
        use_container_width=True,
    )

if submitted:
    raw_input = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        # Penting: nama kolom harus sama dengan dataset/notebook.
        "day_of_week": day_of_week,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }

    try:
        model_input = prepare_input(raw_input)
        score = float(model.predict(model_input, verbose=0).ravel()[0])
        prediction_yes = score >= threshold

        st.divider()
        st.markdown("## Hasil Prediksi")

        score_col, decision_col = st.columns([1, 1.15], gap="large")

        with score_col:
            with st.container(border=True):
                st.metric(
                    "Skor probabilitas model",
                    f"{score * 100:.2f}%"
                )
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score * 100,
                        number={"suffix": "%", "font": {"size": 28}},
                        title={"text": "Skor prediksi"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1769aa"},
                            "steps": [
                                {"range": [0, threshold * 100], "color": "#fee2e2"},
                                {"range": [threshold * 100, 100], "color": "#dcfce7"},
                            ],
                            "threshold": {
                                "line": {"color": "#b91c1c", "width": 3},
                                "thickness": 0.8,
                                "value": threshold * 100,
                            },
                        },
                    )
                )
                gauge.update_layout(
                    height=250,
                    margin=dict(l=18, r=18, t=55, b=12),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(gauge, use_container_width=True)

        with decision_col:
            with st.container(border=True):
                st.markdown("### Rekomendasi Kampanye")
                if prediction_yes:
                    st.success("🎯 POTENSIAL – TARGETKAN NASABAH")
                    st.markdown(
                        "Skor model berada pada atau di atas threshold. "
                        "Nasabah dapat diprioritaskan untuk tindak lanjut "
                        "kampanye deposito."
                    )
                else:
                    st.warning("⏸️ PRIORITAS LEBIH RENDAH")
                    st.markdown(
                        "Skor model berada di bawah threshold. "
                        "Pertimbangkan prioritas prospek lain sesuai kapasitas "
                        "dan strategi kampanye."
                    )

                st.write(f"**Threshold keputusan:** {threshold:.2f}")
                st.write(
                    f"**Selisih dari threshold:** "
                    f"{(score - threshold) * 100:+.2f} poin persentase"
                )

                st.caption(
                    "Prediksi merupakan estimasi berdasarkan pola historis "
                    "dataset, bukan jaminan nasabah akan atau tidak akan "
                    "berlangganan deposito."
                )

        with st.expander("Lihat ringkasan input dan fitur hasil rekayasa"):
            st.dataframe(
                pd.DataFrame([raw_input]).T.rename(columns={0: "Nilai input"}),
                use_container_width=True,
            )
            st.write({
                "pdays_contacted": int(pdays != -1),
                "balance_per_age": float(balance / age) if age else None,
            })

    except Exception as error:
        st.error("Prediksi gagal diproses. Periksa input dan artefak model.")
        st.exception(error)

st.divider()
footer_left, footer_right = st.columns(2)
with footer_left:
    st.caption("Deep Learning Project · Bank Marketing UCI Dataset")
with footer_right:
    st.caption("Multi-Layer Perceptron (MLP) · TensorFlow · Streamlit")