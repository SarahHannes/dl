import urllib.request

from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
    """
    Load saved model from source url.
    :return: keras.engine.sequential.Sequential
    """
    urllib.request.urlretrieve(
        # Source url: using gitHub permalink
        'https://github.com/SarahHannes/dl/raw/e1a0ee81c43f69772842187f980694a29b8d19cc/cnn/model/20_bread_weights-improvement-09-0.88.hdf5',
        # Destination path
        'model.hdf5')
    model_path = './model.hdf5'
    model = tf.keras.models.load_model(model_path)
    return model


def get_prediction(img):
    """
    Get class label and confidence level for user uploaded image.
    :param image: Uploaded image file
    :return: str, float
    """
    class_names = ['good', 'moldy']
    model = load_model()
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = tf.image.resize(img_array, [180, 180])
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    return [predicted_class, 100 * np.max(score)]


st.set_page_config(page_title='Bready', page_icon='🍴')
st.markdown("<h1 style='text-align: center; color: grey;'>To eat or not to eat... 🍞 🥐 🥖</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload image to start!", type='jpg')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    st.write("")
    prediction = get_prediction(image)
    prediction[0] = prediction[0].title()
    if prediction[0] == 'Moldy':
        color = '#800000'
    else:
        color = '#008080'

    html_str = f"""
    <h2 style="text-align: center;">Predicted class: <span style="color: {color};">{prediction[0]}</span></h2>
    <h3 style="text-align: center;"><span style="color: #808080;">Confidence level: {round(prediction[1], 1)}%</span></h3>
    """

    st.markdown(html_str, unsafe_allow_html=True)