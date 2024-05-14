#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data_loading.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")

# data_preprocessing.py
import pandas as pd

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['GENDER'])
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data[['AGE', 'SYMPTOM_COUNT']] = scaler.fit_transform(data[['AGE', 'SYMPTOM_COUNT']])
    
    return data

# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(data):
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1

# streamlit_app.py
import streamlit as st
import joblib

def main():
    st.title("Lung Cancer Prediction")
    
    # Load trained model
    model = joblib.load('model.pkl')
    
    # Input data
    age = st.slider("Age", min_value=1, max_value=100, value=50, step=1)
    smoking = st.selectbox("Smoking", [1, 2])  # Assuming 1: Yes, 2: No
    yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])  # Assuming 1: Yes, 2: No
    anxiety = st.selectbox("Anxiety", [1, 2])  # Assuming 1: Yes, 2: No
    peer_pressure = st.selectbox("Peer Pressure", [1, 2])  # Assuming 1: Yes, 2: No
    chronic_disease = st.selectbox("Chronic Disease", [1, 2])  # Assuming 1: Yes, 2: No
    fatigue = st.selectbox("Fatigue", [1, 2])  # Assuming 1: Yes, 2: No
    allergy = st.selectbox("Allergy", [1, 2])  # Assuming 1: Yes, 2: No
    wheezing = st.selectbox("Wheezing", [1, 2])  # Assuming 1: Yes, 2: No
    alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2])  # Assuming 1: Yes, 2: No
    coughing = st.selectbox("Coughing", [1, 2])  # Assuming 1: Yes, 2: No
    shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2])  # Assuming 


# In[ ]:




