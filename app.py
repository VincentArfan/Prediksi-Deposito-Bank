import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="Prediksi Deposito Bank", page_icon="🏦", layout="wide")

st.title("🏦 Prediksi Kampanye Pemasaran Bank (MLP)")
st.write("Aplikasi untuk memprediksi apakah nasabah akan berlangganan deposito berjangka.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('mlp_bank_marketing.h5')
    scaler = joblib.load('scaler.pkl')
    kolom_fitur = joblib.load('kolom_fitur.pkl')
    return model, scaler, kolom_fitur

model, scaler, kolom_fitur = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (Age)", min_value=18, max_value=100, value=35)
    job = st.selectbox("Pekerjaan (Job)", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], index=4)
    marital = st.selectbox("Status Pernikahan", ['divorced', 'married', 'single', 'unknown'], index=2)
    education = st.selectbox("Pendidikan", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'tertiary', 'unknown'], index=6)
    default = st.radio("Punya Kredit Macet? (Default)", ['no', 'yes', 'unknown'], index=0)
    balance = st.number_input("Saldo Tahunan (Balance)", value=1500)
    housing = st.radio("Cicilan Rumah (Housing)", ['no', 'yes', 'unknown'], index=1)
    loan = st.radio("Pinjaman Pribadi (Loan)", ['no', 'yes', 'unknown'], index=0)

with col2:
    contact = st.selectbox("Tipe Kontak", ['cellular', 'telephone'], index=0)
    day_of_week = st.number_input("Hari Kontak Terakhir (Tanggal)", min_value=1, max_value=31, value=15)
    month = st.selectbox("Bulan Kontak Terakhir", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
    campaign = st.number_input("Jumlah Kontak Kampanye Ini", min_value=1, value=2)
    pdays = st.number_input("Hari Sejak Kontak Terakhir (pdays: -1 jika belum pernah)", value=-1)
    previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", value=0)
    poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ['failure', 'nonexistent', 'success', 'unknown'], index=3)

if st.button("Prediksi Sekarang", type="primary"):
    data_input = {
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
        'default': [default], 'balance': [balance], 'housing': [housing], 
        'loan': [loan], 'contact': [contact], 'day_of_week': [day_of_week], 
        'month': [month], 'campaign': [campaign], 'pdays': [pdays], 
        'previous': [previous], 'poutcome': [poutcome]
    }
    df_in = pd.DataFrame(data_input)
    df_encoded = pd.get_dummies(df_in)
    df_encoded = df_encoded.reindex(columns=kolom_fitur, fill_value=0)
    df_scaled = scaler.transform(df_encoded)
    
    prob = model.predict(df_scaled)[0][0]
    threshold = 0.5
    
    st.subheader("Hasil Prediksi:")
    if prob > threshold:
        st.success(f"**YES (Berlangganan)** — Probabilitas: {prob*100:.2f}%")
    else:
        st.error(f"**NO (Tidak Berlangganan)** — Probabilitas: {prob*100:.2f}%")
