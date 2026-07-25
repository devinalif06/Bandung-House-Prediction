import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "clean_df.csv"

st.set_page_config(page_title="Prediksi Harga Rumah Bandung Raya", page_icon="🏠")

@st.cache_resource
def load_model(_mtime: float):
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    locations = sorted(df["Location"].unique().tolist())
    cities = sorted(df["City/Regency"].unique().tolist())
    return locations, cities

model = load_model(MODEL_PATH.stat().st_mtime)
locations, cities = load_reference_data()

if not hasattr(model, "named_steps"):
    st.error(
        f"Model yang ter-load bukan Pipeline lengkap (tipe: {type(model).__name__}). "
        f"Cek ulang file di {MODEL_PATH} -- pastikan hasil joblib.dump(final_model, ...), "
        f"bukan model mentah."
    )
    st.stop()


st.title("Prediksi Harga Rumah Bandung Raya")
st.caption(
    "Model regresi (Extra Trees Regressor) untuk mengestimasi harga rumah "
    "di Kota Bandung, Kabupaten Bandung, dan Kabupaten Bandung Barat."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("Kota/Kabupaten", cities)
    location = st.selectbox("Lokasi/Kecamatan", locations)
    bedroom = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
    bathroom = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2, step=1)

with col2:
    carport = st.number_input("Jumlah Carport", min_value=0, max_value=5, value=1, step=1)
    land = st.number_input("Luas Tanah (m²)", min_value=10, max_value=2000, value=100, step=5)
    building = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=2000, value=90, step=5)

st.divider()

if st.button("Prediksi Harga", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "Location": [location],
        "City/Regency": [city],
        "Bedroom": [bedroom],
        "Bathroom": [bathroom],
        "Carport": [carport],
        "Land": [land],
        "Building": [building],
    })

    pred_log = model.predict(input_data)
    predicted_price = np.expm1(pred_log)[0]

    st.success(f"### Estimasi Harga: Rp {predicted_price:,.0f}")
    st.caption(
        "Estimasi ini berdasarkan model dengan MAPE ~18% pada data uji — "
        "gunakan sebagai referensi kasar, bukan valuasi resmi."
    )

st.divider()
st.caption(
    "Dibangun dengan scikit-learn & Streamlit. "
    "[Lihat notebook & metodologi lengkap di GitHub](#)"
)
