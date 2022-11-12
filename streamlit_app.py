# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import maissajalon3 as md

st.title("welcome to my app")
text=st.text_input("enter your comment")
number = st.number_input('Insert a number of topics',min_value=1,max_value=15,step=1)

p,b=md.prediction(text, number, md.dict)
if st.button('Predict'):
    st.write(p)
    st.write(b)

