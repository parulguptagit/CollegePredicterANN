import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('College Admissions prediction')

greScore=st.slider('GRE Score', 260,340)
toeflScore=st.slider('TOEFL Score', 0,120)
universityRating= st.selectbox("Uni Rating", [1,2,3,4,5])
sop= st.number_input("SOP")
lor=st.number_input("LOR")
cgpa=st.number_input("CGPA")
research=st.selectbox("Research",[0,1])

input_data = pd.DataFrame({
    'GREScore': [greScore],
    'TOEFLScore': [toeflScore],
    'UniversityRating':[universityRating],
    'SOP':[sop],
    "LOR":[lor],
    'CGPA':[cgpa],
    'Research':[research]
})

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

st.write('Acceptance probability', prediction_probab)