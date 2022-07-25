# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:32:17 2022

@author: intan
"""

import pickle
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH=os.path.join(os.getcwd(),'model','best_estimator.pkl')
with open(MODEL_PATH,'rb') as file:
    model=pickle.load(file)
    
#%%

# new_data=[65,158,2.3,3,0,0,1]

# print(model.predict(np.expand_dims(new_data,axis=0)))

#%%


import streamlit as st
import joblib
import pandas as pd
from PIL import Image

st.set_page_config( page_title="Heart Attack Prediction App", page_icon=":heart:", layout="wide")

col1, col2 = st.columns(2)

with col1:
    st.title('Heart Attack Predictor App')
    st.write('This app is built using data from Heart Attack Analysis & Prediction Dataset by Kashik Rahman. This is a sample application and cannot be used as a substitute for real medical advice.')
    video_file = open('heart.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    image = Image.open('sign.jpg')
    st.image(image)
    
    my_expander = st.expander(label='All the datasets, infographics and video credit to')
    with my_expander:
        st.write("""
                 https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
                
                 https://youtu.be/_WXxET_DzE4
                 
                 https://www.facebook.com/1975681522647933/photos/a.1975709505978468/2040202582862493/?type=3
                 
                 https://www.facebook.com/christusstmichael/photos/heart-attacks-can-happen-at-any-time-at-christus-st-michael-health-system-and-ch/10157215219449701/?_rdr
                 
                 https://www.modernheartandvascular.com/conditions/heart-attack/
                """)
    
with col2:
    st.subheader('Please fill in the details of the person under consideration and click on the button below!')
    with st.form("Heart Attack Predictor App"):
        age =           st.number_input("Age in Years", 1, 150, 25, 1)
        thalachh =      st.slider('Maximum heart rate achieved', 50, 220, 1)
        oldpeak =       st.slider('ST depression induced by exercise relative to rest', 0.0, 10.0, 0.01)
        cp =            st.selectbox('Chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic',(0,1,2,3))
        exng =          st.selectbox('Exercise induced angina (1 = yes; 0 = no)',(0,1))
        caa =           st.selectbox('Number of major vessels (0-3) colored by flourosopy',(0,1,2,3))
        thall =         st.selectbox('Thallium stress test(2 = normal; 1 = fixed defect; 3 = reversable defect)',(1,2,3))
        
        row = [age, thalachh, oldpeak, cp, exng, caa, thall]
            
        
        # Every form must have a submit button.
        submitted = st.form_submit_button("Analyse")
        if submitted:
            new_data=np.expand_dims(row,axis=0)
            outcome=model.predict(new_data)[0]
            
            if outcome==0:
                st.subheader("You're healthy! Keep it up!!")
                image1 = Image.open('healthy.jpg')
                st.image(image1)
            else:
                st.subheader('Unfortunately, from our database you are at a higher risk of getting a heart attack.')
                st.subheader("But don't freak out! Here are the things you can do to lower the risk:")
                image2 = Image.open('prevention.png')
                st.image(image2)
           
            





