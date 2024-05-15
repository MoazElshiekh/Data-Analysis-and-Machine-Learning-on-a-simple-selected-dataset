#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import joblib
import streamlit as st

# Load the dataset
df = pd.read_csv("SurveyLungCancer.csv")


# Describe the dataset and its features
print("\n1.SurveyLungCancer :")
print(df.info())


# In[4]:


# Data Preprocessing
# Handling Missing Values, Outliers, and Inconsistencies
# No missing values, outliers, or inconsistencies found in this dataset

# Data Preparation
# Encoding Categorical Variables
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])


# In[6]:


# Feature Engineering
# Adding Symptom Count
def calculate_symptom_count(row):
    symptom_columns = df.columns[2:-1]
    return row[symptom_columns].sum()

df['Symptom Count'] = df.apply(calculate_symptom_count, axis=1)

# Saving the updated dataset back to a CSV file
df.to_csv("SurveyLungCancer_with_symptom_countnew.csv", index=False)


# In[7]:



# Data Visualization
# Histograms for Numerical Features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Count plots for Categorical Variables
plt.figure(figsize=(10, 6))
sns.countplot(x='GENDER', data=df)
plt.title('Count of Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='LUNG_CANCER', data=df)
plt.title('Count of Lung Cancer')
plt.show()

# Pairplot for Numerical Features
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[8]:


# Preparing Data for Model Training
# Splitting the dataset into features (X) and target variable (y)
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Training and Evaluating Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall


# In[ ]:




