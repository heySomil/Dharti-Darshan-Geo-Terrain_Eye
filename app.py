import streamlit as st
from PIL import Image

import numpy as np
import tensorflow as tf

from load import init
from config import idx2class

# Set page width
st.set_page_config(layout="wide")

# Custom CSS for styling
st.write(
    """
    <style>
    body {
        text-align: center;
        background-color: #f4f4f4;
        color: #333;
    }

    .title {
        font-size: 1.5em;
        color: #4CAF50;
    }

    .file-upload {
        padding: 20px;
        background-color: #ffffff;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
    }

    .image-preview {
        margin-top: 20px;
    }

    .prediction {
        font-size: 1.5em;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title
st.title('Dharti Darshan - Geo Terrain Eye')
st.markdown('<p class="title">Just upload the image and get to know what terrain it has.</p>', unsafe_allow_html=True)

file_type = 'jpg'
uploaded_file = st.file_uploader("Upload a file", type=file_type, key="file_uploader")

# Load the model
global model
model = init()
st.text('Loaded pretrained Model')

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    image = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr).argmax()
    
    # Display prediction
    st.markdown('<p class="prediction">Predicted Class: {0}</p>'.format(idx2class[prediction]), unsafe_allow_html=True)
