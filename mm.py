#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")

# Prepare the data
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save the trained model
with open('classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Load the trained model
with open('classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[1]:


import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Preprocess the input data (convert categorical variables to numerical, etc.)
    # For simplicity, I assume that the input data is already preprocessed
    input_data = [
        data["age"],
        data["smoking"],
        data["yellow_fingers"],
        data["anxiety"],
        data["peer_pressure"],
        data["chronic_disease"],
        data["fatigue"],
        data["allergy"],
        data["wheezing"],
        data["alcohol_consuming"],
        data["coughing"],
        data["shortness_of_breath"],
        data["swallowing_difficulty"],
        data["chest_pain"],
    ]

    # Make a prediction using the trained model
    prediction = classifier.predict([input_data])

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)


# In[5]:


data = pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")


# In[6]:


import joblib

# Save the trained model to a file
joblib.dump(classifier, "classifier.joblib")


# In[13]:


import streamlit as st

with open("moazzz") as f:
    html_content = f.read()

st.markdown(html_content, unsafe_allow_html=True)


# In[14]:


import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model and the scaler
model = pickle.load(open('classifier.pkl', 'rb'))

# Define a function to preprocess the user input data
def preprocess_data(data):
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data)
    # Scale the data using the scaler
    data = scaler.transform(data)
    return data

# Define a function to predict the probability of lung cancer
def predict_lung_cancer(data):
    # Preprocess the data
    data = preprocess_data(data)
    # Predict the probability of lung cancer
    probability = model.predict_proba(data)
    return probability

# Create a user interface for the user to input their symptoms
st.title("Lung Cancer Prediction Model")
st.write("Please enter your symptoms below:")
age = st.number_input("Age", min_value=18, max_value=100)
gender = st.radio("Gender", ["Male", "Female"])
smoking_history = st.slider("Smoking History (in years)", min_value=0, max_value=50)
symptom1 = st.checkbox("Symptom 1")
symptom2 = st.checkbox("Symptom 2")
symptom3 = st.checkbox("Symptom 3")
# Add more symptoms as needed

# Prepare the user input data
user_data = {
    "age": [age],
    "gender": [gender],
    "smoking_history": [smoking_history],
    "symptom1": [symptom1],
    "symptom2": [symptom2],
    "symptom3": [symptom3],
    # Add more symptoms as needed
}
user_data = pd.DataFrame(user_data)

# Predict the probability of lung cancer
if st.button("Predict"):
    probability = predict_lung_cancer(user_data)
    st.write("The probability of lung cancer is:", round(probability[0][1] * 100, 2), "%")


# In[ ]:




