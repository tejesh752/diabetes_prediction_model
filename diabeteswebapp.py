# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:50:27 2024

@author: TEJESH VARMA
"""

import numpy as  np
import pickle
import streamlit as st

loaded_model = pickle.load(open("trained_model.sav",'rb'))

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    st.title("Diabetes prediction")
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Bloodpressure level")
    SkinThickness=st.text_input("skin thickness value")
    Insulin=st.text_input("insulin level")
    BMI=st.text_input("BMI value")
    DiabetesPedigreeFunction=st.text_input("Diabetes pedigree function value")
    Age=st.text_input("Age of the person")
    
    diagnosis = ""
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
    
    
    
if __name__=='__main__':
    main()    
    
