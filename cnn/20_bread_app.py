# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:19:19 2022

@author: Sarah
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
import urllib.request

# Download best model
@st.cache(allow_output_mutation=True)
def load_model():
        urllib.request.urlretrieve(
                # Source url: using github permalink
                'https://github.com/SarahHannes/dl/raw/e1a0ee81c43f69772842187f980694a29b8d19cc/cnn/model/20_bread_weights-improvement-09-0.88.hdf5',
                # Destination path
                'model.hdf5')
        MODEL_PATH = './model.hdf5'
        model = tf.keras.models.load_model(MODEL_PATH)
        return model

# Upload image
# Predict image
# Plot image

st.write('Good Bread Moldy Bread classifier 🍞🥐🥖')
img = st.file_uploader("Upload your bread image!")
st.image(img)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Predict using loaded model
predictions = model3.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Plot user input image
plt.imshow(img_array[0].numpy().astype("uint8"))
plt.title("Predicted class: {} ({:.2f}% confidence)".format(class_names[np.argmax(score)], 100 * np.max(score)))
plt.axis('off')
plt.show()
