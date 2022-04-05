# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:19:19 2022

@author: Sarah
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
import h5py

# Download best model
# download_from = "https://github.com/SarahHannes/dl/raw/main/cnn/model/20_bread_weights-improvement-09-0.88.hdf5"
# model_f = h5py.File(download_from, 'r')

# current_path = os.getcwd()
# st.write('current path', current_path)
# current_path = os.chdir("")
# st.write('current path',  )
model_path = os.path.dirname('cnn/model/20_bread_weights-improvement-09-0.88.hdf5')
model = tf.keras.models.load_model(model_path)
st.write(model.summary())

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
