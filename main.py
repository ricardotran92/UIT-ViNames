import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'FeMale', '1': 'Male'}

st.title('Guess Gender via Name')

image = Image.open('Gender via Name.jpg')
st.image(image)

input_ec = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vinames.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a (full) name')
txt = st.text_area('', '')

if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text(class_list[label])
