import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Memuat model
model = load_model('effnet.h5')

# Fungsi untuk prediksi gambar
def img_pred(image):
    img = Image.open(image)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]

    if p == 0:
        label = 'Tuberkulosis'
        description = 'Model memprediksi bahwa gambar menunjukkan gejala yang konsisten dengan Tuberkulosis.'
    else:
        label = 'Normal'
        description = 'Model memprediksi bahwa gambar tidak menunjukkan gejala Tuberkulosis.'

    return label, description, p

# Judul halaman
st.title("Prediksi Penyakit Tuberkulosis")

# Deskripsi aplikasi
st.markdown("""
          Di halaman ini, Unggah gambar Image X-Ray diprediksi apakah menunjukkan gejala Tuberkulosis atau tidak.
          """)

# Seksi unggah gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Melakukan prediksi
    label, description, probability = img_pred(uploaded_file)
    
    # Menampilkan hasil prediksi dalam tabel
    st.subheader("Hasil Prediksi")
    # Menggunakan CSS untuk menengahkan teks
    st.markdown(f'<p style="text-align:center; font-size:20px;">Label: {label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; font-size:16px;">Deskripsi: {description}</p>', unsafe_allow_html=True)
    st.write("")  # Menambahkan baris kosong
    
    # Menampilkan gambar dengan label prediksi
    st.subheader("Visualisasi Prediksi")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Gambar asli
    ax1.imshow(np.array(image))
    ax1.set_title('Gambar Asli')
    ax1.axis('off')
    
    # Gambar dengan label prediksi
    ax2.imshow(np.array(image))
    ax2.set_title(f'Prediksi: {label}')
    ax2.text(0.5, -0.1, description, fontsize=12, ha='center', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    ax2.axis('off')
    
    st.pyplot(fig)

    # Visualisasi grafik interaktif menggunakan Plotly
    st.subheader("Grafik Probabilitas")
    labels = ['Tuberkulosis', 'Normal']
    probabilities = [1 - probability, probability]  # Invert probability for 'Tuberkulosis'
    colors = ['lightcoral', 'lightskyblue']

    fig = go.Figure(data=[go.Bar(x=labels, y=probabilities, marker_color=colors)])
    fig.update_layout(title='Probabilitas Prediksi', xaxis_title='Label', yaxis_title='Probabilitas')
    st.plotly_chart(fig)

# Informasi tambahan tentang model
st.markdown("---")
st.subheader("Informasi tentang Model")
st.write("- Prediksi Image X-Ray | Model EfficientNetB0")
st.write("- Tugas Bussines Data Engineering | Toviek Hidayat")
st.write("- UR Juli 2024")
