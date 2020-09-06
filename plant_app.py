import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import tensorflow as tf


image_size = 64
st.set_option('deprecation.showfileUploaderEncoding', False)
labels = ['Pepper__bell___Bacterial_spot',
          'Pepper__bell___healthy',
          'Potato___Early_blight',
          'Potato___Late_blight',
          'Potato___healthy',
          'Tomato_Bacterial_spot',
          'Tomato_Early_blight',
          'Tomato_Late_blight',
          'Tomato_Leaf_Mold',
          'Tomato_Septoria_leaf_spot', 
          'Tomato_Spider_mites_Two_spotted_spider_mite', 
          'Tomato__Target_Spot', 
          'Tomato__Tomato_YellowLeaf__Curl_Virus',
          'Tomato__Tomato_mosaic_virus', 
          'Tomato_healthy']


def main():
    st.title("Plant Disease Classification")
    st.header("Classify Your Custom Images")
    st.text("Upload a picture for clasification into healthy or diseased plant")
    uploaded_file = st.file_uploader("Choose a plant picture", type="jpg")
    if uploaded_file is not None:
        model = tf.keras.models.load_model('fullTrain.h5')
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Picture', use_column_width=True)
        size = (64, 64)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
        data[0] = normalized_image_array
        preds = model.predict(data)
        label = labels[np.argmax(preds)]
        st.write("The disease is:"+ label)
        


    
if __name__ == '__main__':
    main()
    