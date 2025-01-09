import streamlit as st
import pandas as pd
import numpy as np
import sklearn 
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder=LabelEncoder()
scaler=StandardScaler()



# load model and data
model=pickle.load(open('01_logistic_model.pkl','rb'))




# create web app .......
st.title('Logistic Regression for Churn Prediction')
gender=st.selectbox('Select Gender :',options=['Female','Male'])
SeniorCitizen=st.selectbox('You are a senior citizen ? ',options=['Yes','No'])
Partner=st.selectbox('Do you have a partner ?',options=['Yes','No'])
Dependents=st.selectbox('Are you dependends on other ?', options=['Yes','No'])
tenure=st.text_input('Enter your tenure?')
PhoneService=st.selectbox('Do you have phone service ?', options=['Yes','No'])
MultipleLines=st.selectbox('Do you have multilines service ?',options=['Yes','No','No phone service'])
Contract=st.selectbox('Your contract ?',options=['One year','Two year','Month-to-month'])
TotalCharges=st.text_input('Enter your total charges ?')









# helper function..........

def predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges):
    data={
        'gender':[gender],
        'SeniorCitizen':[SeniorCitizen],
        'Partner':[Partner],
        'Dependents':[Dependents],
        'tenure':[tenure],
        'PhoneService':[PhoneService],
        'MultipleLines':[MultipleLines],
        'Contract':[Contract],
        'TotalCharges':[TotalCharges]
    }
    df1=pd.DataFrame(data)
    
  

    # Encode the categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df1[column] = label_encoder.fit_transform(df1[column])
    df1 = scaler.fit_transform(df1)

    result = model.predict(df1).reshape(1,-1)
    return result[0]


# button
if st.button('predict'):
    result=predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges)

    if result==0:
        st.write('Not Churn')
    else:
        st.write('Churn')