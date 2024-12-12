import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Fungsi untuk membersihkan kolom dari karakter non-standar dan whitespace tambahan
def clean_column(column):
    """Membersihkan kolom dari karakter non-standar dan whitespace tambahan."""
    return column.astype(str).str.replace(r'\xa0', '', regex=True).str.strip()

# Fungsi untuk membersihkan dan mengonversi data menjadi numerik
def clean_and_convert_data(data):
    # Mengonversi kolom numerik dan menghapus karakter non-standar
    data['Usia'] = clean_column(data['Usia']).str.replace(' bulan', '').astype(float)
    data['Berat'] = pd.to_numeric(data['Berat'], errors='coerce')
    data['Tinggi'] = pd.to_numeric(data['Tinggi'], errors='coerce')
    data['ZS_BB_TB'] = pd.to_numeric(data['ZS_BB_TB'], errors='coerce')

    # Hapus baris dengan nilai kosong setelah konversi
    data.dropna(subset=['Usia', 'Berat', 'Tinggi', 'ZS_BB_TB'], inplace=True)

    return data

# Judul Aplikasi
st.title("Prediksi Status Gizi Anak")

# Memuat Model dan Label Encoder
try:
    model = joblib.load('xgboost_best_model.pkl')  # Model yang telah dilatih
    label_encoder = joblib.load('label_encoder_jk.pkl')  # Label Encoder
    st.success("Model dan Label Encoder berhasil dimuat!")
except FileNotFoundError as e:
    st.error(f"File model atau encoder tidak ditemukan: {e}")
    st.stop()

# Input File CSV
# uploaded_file = st.file_uploader("Upload file CSV untuk prediksi", type=["csv"])

# if uploaded_file:
#     # Membaca File CSV dengan encoding yang benar
#     try:
#         data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # Coba gunakan 'ISO-8859-1' atau 'utf-8'
#         st.write("Data yang diunggah:")
#         st.dataframe(data)

#         # Validasi Kolom yang diperlukan
#         required_columns = ['JK', 'Usia', 'Berat', 'Tinggi', 'ZS_BB_TB']  # Kolom yang dibutuhkan
#         if all(col in data.columns for col in required_columns):
#             try:
#                 # Membersihkan kolom yang diperlukan
#                 data['JK'] = clean_column(data['JK'])
#                 data = clean_and_convert_data(data)  # Bersihkan dan konversi kolom menjadi numerik

#                 # Encoding untuk jenis kelamin (JK)
#                 data['JK'] = label_encoder.transform(data['JK'])  # Label encoding untuk kolom 'JK'

#                 # Prediksi
#                 input_data = data[required_columns].values
#                 predictions = model.predict(input_data)

#                 # Menambahkan Prediksi ke DataFrame
#                 data['Prediksi'] = predictions
#                 st.write("Hasil Prediksi:")
#                 st.dataframe(data)

#                 # Unduh Hasil Prediksi
#                 st.download_button(
#                     label="Unduh Hasil Prediksi",
#                     data=data.to_csv(index=False),
#                     file_name="hasil_prediksi.csv",
#                     mime="text/csv"
#                 )
#             except Exception as e:
#                 st.error(f"Kesalahan saat memproses data: {e}")
#         else:
#             st.error(f"Kolom yang diperlukan: {', '.join(required_columns)}")
#     except Exception as e:
#         st.error(f"Kesalahan saat membaca file: {e}")

# else:
st.write("Atau masukkan data secara manual:")

    # Input Manual
JK = st.selectbox("Jenis Kelamin", options=['Laki-laki', 'Perempuan'])
Usia = st.number_input("Usia (dalam bulan)", min_value=0, max_value=120, step=1)
Berat = st.number_input("Berat (kg)", min_value=0.0, max_value=100.0, step=0.1)
Tinggi = st.number_input("Tinggi (cm)", min_value=0.0, max_value=200.0, step=0.1)
ZS_BB_TB = st.number_input("Z-Score Berat Badan terhadap Tinggi", step=0.01)  # Tambahkan fitur

    # Prediksi dari Input Manual
if st.button("Prediksi"):
    try:
            # Konversi 'JK' ke numerik (Label Encoding)
        JK = 1 if JK == 'Laki-laki' else 0  # 1 untuk Laki-laki, 0 untuk Perempuan
            
            # Membuat input data untuk prediksi
        input_data = np.array([[JK, Usia, Berat, Tinggi, ZS_BB_TB]])  # Tambahkan fitur ZS_BB_TB
        prediction = model.predict(input_data)
        
            # Menampilkan hasil prediksi
        st.success(f"Hasil Prediksi: {prediction[0]}")
    except Exception as e:
        st.error(f"Kesalahan saat prediksi: {e}")

st.write('Keterangan Status Gizi:')
st.write('0 = Gizi Baik')
st.write('1 = Gizi Buruk')
st.write('2 = Gizi Kurang')
st.write('3 = Gizi Lebih')
st.write('4 = Obesitas')
st.write('5 = Resiko Gizi Lebih')