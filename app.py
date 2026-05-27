import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

try:
    model = load_model("colon_model2.keras")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

labels = ['kanker', 'normal']

def crop_image(image):
    width, height = image.size
    new_width = min(width, height)
    new_height = min(width, height)
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    return image.crop((left, top, right, bottom))

def classify_image(image):
    image = image.convert("RGB")

    cropped_image = crop_image(image)
    img = cropped_image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probability = float(predictions[0][0])

    if probabilitas_kanker > 0.5:
        kelas_prediksi = 'normal'
        skor_keyakinan = probability  
    else:
        kelas_prediksi = 'kanker'
        skor_keyakinan = 1.0 - probability
        
    # max_index = np.argmax(predictions[0])
    return kelas_prediksi, skor_keyakinan


st.title("Klasifikasi Histopatologi Kanker Usus Besar")
uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:    
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if st.button("Klasifikasikan"):
        with st.spinner("Memproses..."):
            label, confidence = classify_image(image)

            st.write("---")
            if label == 'kanker':
                st.error(f"Hasil Prediksi: **{label.upper()}**")
            else:
                st.success(f"Hasil Prediksi: **{label.upper()}**")

            st.info(f"Keyakinan Akurasi Model: {confidence * 100:.2f}")
