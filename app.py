import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CUSTOM CSS (SaaS UI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Deposito Bank | MLP Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .hero-header {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        padding: 26px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(13, 110, 253, 0.15);
    }
    .hero-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 1.8rem;
    }
    .hero-header p {
        color: #e0e7ff;
        margin-top: 6px;
        margin-bottom: 0;
        font-size: 0.95rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #198754 0%, #157347 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(25, 135, 84, 0.2);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #157347 0%, #146c43 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(25, 135, 84, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. LOAD ARTIFACTS (FOLDER `artifacts/`)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
  try:
    model = tf.keras.models.load_model("artifacts/bank_marketing_mlp.keras")
    scaler = joblib.load("artifacts/scaler.pkl")
    kolom_fitur = joblib.load("artifacts/feature_columns.pkl")

    # Memuat metadataOpsional jika tersedia
    try:
      with open("artifacts/model_info.json", "r") as f:
        model_info = json.load(f)
    except Exception:
      model_info = None

    return model, scaler, kolom_fitur, model_info
  except Exception as e:
    st.error(
        f"⚠️ Gagal memuat file artefak dari folder `artifacts/`. Detail: {e}"
    )
    st.stop()


model, scaler, kolom_fitur, model_info = load_artifacts()

# ---------------------------------------------------------
# 3. SIDEBAR PANEL
# ---------------------------------------------------------
with st.sidebar:
  st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
  st.title("Bank Analytics")
  st.caption("Multi-Layer Perceptron (MLP)")
  st.markdown("---")

  st.subheader("⚙️ Konfigurasi Model")
  threshold = st.slider(
      "Threshold Probabilitas",
      min_value=0.10,
      max_value=0.90,
      value=0.40,
      step=0.05,
  )
  st.info("💡 Menyeimbangkan Precision & Recall kelas minoritas (Deposit=Yes).")

  st.markdown("---")
  st.markdown("**Status Artefak Directory:**")
  st.success("✅ `bank_marketing_mlp.keras` Loaded")
  st.success("✅ `scaler.pkl` Loaded")
  st.success(f"✅ `feature_columns.pkl` ({len(kolom_fitur)} Fitur)")
  if model_info:
    st.success("✅ `model_info.json` Loaded")

# ---------------------------------------------------------
# 4. HEADER UTAMA
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero-header">
    <h1>🏦 AI Bank Marketing Deposit Predictor</h1>
    <p>Aplikasi Prediksi Deposito Berjangka Berbasis Deep Learning Multi-Layer Perceptron (MLP)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 5. FORM INPUT DATA NASABAH
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
  with st.container(border=True):
    st.subheader("👤 Profil Nasabah")
    age = st.number_input("Umur (Age)", min_value=18, max_value=100, value=35)
    job = st.selectbox(
        "Pekerjaan (Job)",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
        index=4,
    )
    marital = st.selectbox(
        "Status Pernikahan", ["divorced", "married", "single"], index=1
    )
    education = st.selectbox(
        "Pendidikan", ["primary", "secondary", "tertiary", "unknown"], index=2
    )
    default = st.radio(
        "Punya Kredit Macet? (Default)", ["no", "yes"], index=0, horizontal=True
    )
    balance = st.number_input("Saldo Tahunan dalam EUR (Balance)", value=1500)
    housing = st.radio(
        "Kredit Rumah? (Housing)", ["no", "yes"], index=1, horizontal=True
    )
    loan = st.radio(
        "Pinjaman Pribadi? (Loan)", ["no", "yes"], index=0, horizontal=True
    )

with col_right:
  with st.container(border=True):
    st.subheader("📞 Riwayat Kontak Kampanye")
    contact = st.selectbox(
        "Tipe Kontak", ["cellular", "telephone", "unknown"], index=0
    )
    day = st.number_input(
        "Hari Kontak Terakhir (Tanggal 1-31)",
        min_value=1,
        max_value=31,
        value=15,
    )
    month = st.selectbox(
        "Bulan Kontak Terakhir",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        index=4,
    )
    campaign = st.number_input(
        "Jumlah Kontak Selama Kampanye Ini", min_value=1, value=2
    )
    pdays = st.number_input(
        "Hari Sejak Kontak Kampanye Lalu (pdays: -1 jika belum pernah)",
        value=-1,
    )
    previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", value=0)
    poutcome = st.selectbox(
        "Hasil Kampanye Sebelumnya",
        ["failure", "other", "success", "unknown"],
        index=3,
    )

st.write("")

# ---------------------------------------------------------
# 6. PROSES INFERENSI PREDIKSI
# ---------------------------------------------------------
if st.button("🔮 Eksekusi Prediksi Nasabah", use_container_width=True):
  data_input = {
      "age": [age],
      "job": [job],
      "marital": [marital],
      "education": [education],
      "default": [default],
      "balance": [balance],
      "housing": [housing],
      "loan": [loan],
      "contact": [contact],
      "day": [day],
      "month": [month],
      "campaign": [campaign],
      "pdays": [pdays],
      "previous": [previous],
      "poutcome": [poutcome],
  }

  df_in = pd.DataFrame(data_input)

  # Otomatis mengecek jika ada Feature Engineering di notebook Anda
  if "balance_per_age" in kolom_fitur:
    df_in["balance_per_age"] = df_in["balance"] / (df_in["age"] + 1e-5)
  if "pdays_contacted" in kolom_fitur:
    df_in["pdays_contacted"] = (df_in["pdays"] != -1).astype(int)

  # One-Hot Encoding & Penyesuaian Kolom dengan feature_columns.pkl
  df_encoded = pd.get_dummies(df_in)
  df_encoded = df_encoded.reindex(columns=kolom_fitur, fill_value=0).astype(
      float
  )

  # Normalisasi & Inseminasi ke Model MLP
  df_scaled = scaler.transform(df_encoded)
  prob = float(model.predict(df_scaled, verbose=0)[0][0])

  # Output Visualisasi Hasil Prediksi
  st.markdown("---")
  st.subheader("📊 Hasil Evaluasi Prediksi")

  res_col1, res_col2 = st.columns([1.2, 1])

  with res_col1:
    if prob >= threshold:
      st.success("🎉 **REKOMENDASI: TARGETKAN NASABAH (YES)**")
      st.markdown(f"""
            * **Probabilitas Berlangganan:** `{prob*100:.2f}%` (Di atas threshold `{threshold*100:.0f}%`)
            * **Tindakan Sales:** Nasabah ini diprioritaskan untuk dihubungi oleh tim telemarketing karena berpotensi tinggi mengambil deposito berjangka.
            """)
    else:
      st.error("❌ **REKOMENDASI: TIDAK PRIORITAS (NO)**")
      st.markdown(f"""
            * **Probabilitas Berlangganan:** `{prob*100:.2f}%` (Di bawah threshold `{threshold*100:.0f}%`)
            * **Tindakan Sales:** Alokasi tim telemarketing dialihkan ke prospek lain untuk menghemat efisiensi biaya operasional kampanye.
            """)

  with res_col2:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 24, "color": "#1e293b"}},
            title={"text": "Probabilitas Berlangganan", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0d6efd"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, threshold * 100], "color": "#fee2e2"},
                    {"range": [threshold * 100, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)