import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Deposito Bank", page_icon="🏦", layout="wide")

st.title("🏦 Prediksi Kampanye Pemasaran Bank (MLP)")
st.write("Aplikasi untuk memprediksi apakah nasabah akan berlangganan deposito berjangka.")

# Load Model & Artifacts dengan Error Handling
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('mlp_bank_marketing.h5')
        scaler = joblib.load('scaler.pkl')
        kolom_fitur = joblib.load('kolom_fitur.pkl')
        return model, scaler, kolom_fitur
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file artifact (`mlp_bank_marketing.h5`, `scaler.pkl`, atau `kolom_fitur.pkl`). Detail: {e}")
        st.stop()

model, scaler, kolom_fitur = load_artifacts()

# Form Input Data Nasabah
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Profil Nasabah")
    age = st.number_input("Umur (Age)", min_value=18, max_value=100, value=35)
    job = st.selectbox("Pekerjaan (Job)", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], index=4)
    marital = st.selectbox("Status Pernikahan", ['divorced', 'married', 'single', 'unknown'], index=2)
    education = st.selectbox("Pendidikan", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'tertiary', 'unknown'], index=6)
    default = st.radio("Punya Kredit Macet? (Default)", ['no', 'yes', 'unknown'], index=0)
    balance = st.number_input("Saldo Tahunan (Balance)", value=1500)
    housing = st.radio("Cicilan Rumah (Housing)", ['no', 'yes', 'unknown'], index=1)
    loan = st.radio("Pinjaman Pribadi (Loan)", ['no', 'yes', 'unknown'], index=0)

with col2:
    st.subheader("📞 Riwayat Kontak Kampanye")
    contact = st.selectbox("Tipe Kontak", ['cellular', 'telephone', 'unknown'], index=0)
    day = st.number_input("Hari Kontak Terakhir (Tanggal 1-31)", min_value=1, max_value=31, value=15)
    month = st.selectbox("Bulan Kontak Terakhir", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
    campaign = st.number_input("Jumlah Kontak Kampanye Ini", min_value=1, value=2)
    pdays = st.number_input("Hari Sejak Kontak Terakhir (pdays: -1 jika belum pernah)", value=-1)
    previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", value=0)
    poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ['failure', 'other', 'success', 'unknown'], index=3)

st.markdown("---")

# Tombol Eksekusi Prediksi
if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
    # 1. Menyusun Data Input (Gunakan kunci 'day' sesuai dataset asli)
    data_input = {
        'age': [age], 
        'job': [job], 
        'marital': [marital], 
        'education': [education],
        'default': [default], 
        'balance': [balance], 
        'housing': [housing], 
        'loan': [loan], 
        'contact': [contact], 
        'day': [day], 
        'month': [month], 
        'campaign': [campaign], 
        'pdays': [pdays], 
        'previous': [previous], 
        'poutcome': [poutcome]
    }
    
    df_in = pd.DataFrame(data_input)
    
    # 2. Sisipkan Feature Engineering di sini jika ada pada Notebook
    # Contoh (uncomment jika digunakan di notebook):
    # df_in['balance_per_age'] = df_in['balance'] / (df_in['age'] + 1e-5)
    # df_in['pdays_contacted'] = (df_in['pdays'] != -1).astype(int)

    # 3. Encoding & Alignment
    df_encoded = pd.get_dummies(df_in)
    df_encoded = df_encoded.reindex(columns=kolom_fitur, fill_value=0)
    
    # 4. Scaling & Predict
    df_scaled = scaler.transform(df_encoded)
    prob = float(model.predict(df_scaled, verbose=0)[0][0])
    threshold = 0.5
    
    # 5. Tampilan Hasil Prediksi
    st.subheader("📊 Hasil Prediksi")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prob > threshold:
            st.success("🎉 **YES (Berlangganan)** — Nasabah berpotensi tinggi untuk mengambil deposito.")
        else:
            st.error("❌ **NO (Tidak Berlangganan)** — Nasabah cenderung tidak berminat mengambil deposito.")
            
    with res_col2:
        st.metric(label="Probabilitas Berlangganan", value=f"{prob * 100:.2f}%")
        st.progress(prob)