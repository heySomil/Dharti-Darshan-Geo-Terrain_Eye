from flask import Flask, render_template, request
from PIL import Image

import numpy as np
import tensorflow as tf

from load import init
from config import idx2class

global model
model = init()

app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    filename = request.args['filename']
    image = tf.keras.utils.load_img(filename, target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr).argmax()
    return idx2class[prediction]

if __name__ == "__main__":
    app.run(debug=True, port=8000)

           