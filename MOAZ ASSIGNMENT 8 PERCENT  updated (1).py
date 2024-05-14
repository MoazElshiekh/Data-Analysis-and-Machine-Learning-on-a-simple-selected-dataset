#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Describe the dataset and its features.
import pandas as pd

# Load the dataset
SurveyLungCancer_df = pd.read_csv("SurveyLungCancer.csv")
# Describe the datasets and their features
print("\n1.SurveyLungCancer :")
print(SurveyLungCancer_df.info())


# In[5]:



# Display summary statistics of numerical features
print("\nSummary Statistics of Numerical Features:")
print(data.describe())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(data.head())


# In[7]:



# 1. Handling Missing Values
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# No missing values found in this dataset, so no further action is required for handling missing values.

# 2. Handling Outliers
# Display summary statistics to identify potential outliers
print("\nSummary Statistics of Numerical Features:")
print(data.describe())

# No obvious outliers detected in the summary statistics.

# 3. Handling Inconsistencies
# No specific inconsistencies mentioned in the dataset description.
# However, it's important to validate the data and ensure consistency in formatting and representation across features.


# In[8]:


# 3. Handling Inconsistencies
# Check data types
print("\nData Types:")
print(data.dtypes)

# Check unique values of categorical features
categorical_features = data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    print(f"\nUnique values for {feature}:")
    print(data[feature].unique())

# Check range and distribution of numerical features
print("\nSummary Statistics of Numerical Features:")
print(data.describe())


# In[9]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode GENDER
data['GENDER'] = label_encoder.fit_transform(data['GENDER'])

# Encode LUNG_CANCER
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

# Check the updated data types
print(data.dtypes)


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data Visualization
# Pairplot
sns.pairplot(data)
plt.show()

# Histograms for Numerical Features
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Count plot for Categorical Variables
plt.figure(figsize=(10, 6))
sns.countplot(x='GENDER', data=data)
plt.title('Count of Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='LUNG_CANCER', data=data)
plt.title('Count of Lung Cancer')
plt.show()

# Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Histograms for Numerical Features
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Count plots for Categorical Variables
plt.figure(figsize=(10, 6))
sns.countplot(x='GENDER', data=data)
plt.title('Count of Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='LUNG_CANCER', data=data)
plt.title('Count of Lung Cancer')
plt.show()

# Pairplot for Numerical Features
sns.pairplot(data)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[27]:


# Encode 'GENDER' column using one-hot encoding
data = pd.get_dummies(data, columns=['GENDER'], drop_first=True)

# Convert 'LUNG_CANCER' column to binary numeric values
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Check the updated dataframe
print(data.head())


# In[28]:


# Define a function to calculate the symptom count for each individual
def calculate_symptom_count(row):
    # Exclude non-symptom columns (GENDER, AGE, LUNG_CANCER)
    symptom_columns = data.columns[2:-1]  # Exclude GENDER, AGE, and LUNG_CANCER
    # Count the number of symptoms reported for the current individual
    return row[symptom_columns].sum()

# Apply the function to each row to create the new feature "Symptom Count"
data['Symptom Count'] = data.apply(calculate_symptom_count, axis=1)

# Display the updated dataset with the new feature
print(data.head())


# In[29]:


import pandas as pd

# Load the original dataset
data = pd.read_csv("SurveyLungCancer.csv")

# Define a function to calculate the symptom count for each individual
def calculate_symptom_count(row):
    # Exclude non-symptom columns (GENDER, AGE, LUNG_CANCER)
    symptom_columns = data.columns[2:-1]  # Exclude GENDER, AGE, and LUNG_CANCER
    # Count the number of symptoms reported for the current individual
    return row[symptom_columns].sum()

# Apply the function to each row to create the new feature "Symptom Count"
data['Symptom Count'] = data.apply(calculate_symptom_count, axis=1)

# Save the updated dataset back to a CSV file
data.to_csv("SurveyLungCancer_with_symptom_count.csv", index=False)

# Display the updated dataset with the new feature
print(data.head())


# In[45]:


import pandas as pd

# Load the dataset
data = pd.read_csv("SurveyLungCancer_with_symptom_count.csv")

# Map "LUNG_CANCER" column from "YES" and "NO" to 1 and 0 respectively
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Map "GENDER" column from "M" and "F" to 1 and 0 respectively
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})

# Save the updated dataset to a new CSV file
data.to_csv("Updated_SurveyLungCancer_with_symptom_count.csv", index=False)

# Display the first few rows of the updated dataset
print(data.head())


# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
df = pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")

# Split the dataset into features (X) and target variable (y)
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "ROC AUC": roc_auc}

# Display results
results_df = pd.DataFrame(results)
print(results_df)


# In[54]:


# Define the selected model
best_model_name = test_results_df.idxmax(axis=1).iloc[0]
best_model = models[best_model_name]

# Define hyperparameters grid for fine-tuning (example for Logistic Regression)
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model with the best hyperparameters
best_model_tuned = grid_search.best_estimator_
y_pred_tuned = best_model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
roc_auc_tuned = roc_auc_score(y_test, y_pred_tuned)

print("\nPerformance Metrics of the Model after Fine-tuning:")
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("F1 Score:", f1_tuned)
print("ROC AUC Score:", roc_auc_tuned)


# In[58]:


# Load the original dataset (test dataset)
test_df = pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")

# Split the test dataset into features (X_test) and target variable (y_test)
X_test = test_df.drop("LUNG_CANCER", axis=1)
y_test = test_df["LUNG_CANCER"]

# Preprocess the test dataset (apply the same preprocessing steps as done for the training dataset)
# For example, perform imputation, scaling, encoding, etc.

# Make predictions on the preprocessed test dataset using the trained model
y_pred_test = best_model_tuned.predict(X_test)  # Use the best tuned model obtained from previous steps

# Evaluate the performance of the model on the test dataset using appropriate evaluation metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_pred_test)

# Display the performance metrics on the test dataset
print("\nPerformance Metrics on the Test Dataset:")
print("Accuracy:", accuracy_test)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1 Score:", f1_test)
print("ROC AUC Score:", roc_auc_test)


# In[70]:


import joblib

# Save the trained model to a file
joblib.dump(best_model_tuned, "classifier.pkl")


# In[68]:


import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset
data = pd.read_csv("Updated_SurveyLungCancer_with_symptom_count.csv")

# Separate features and target variable
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier (for example, RandomForestClassifier)
classifier = RandomForestClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained classifier as a pickle file
with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)


# In[69]:


import pickle
import streamlit as st

# Load the trained model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Define the function for predicting lung cancer
def predict_lung_cancer(age, smoking, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                        coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    prediction = classifier.predict([[age, smoking, yellow_fingers, anxiety, peer_pressure,
                                      chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                                      coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    if prediction[0] == 1:
        return "Yes"
    else:
        return "No"

def main():
    st.title("Lung Cancer Prediction")
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
    shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2])  # Assuming 1: Yes, 2: No
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2])  # Assuming 1: Yes, 2: No
    chest_pain = st.selectbox("Chest Pain", [1, 2])  # Assuming 1: Yes, 2: No

    if st.button("Predict"):
        result = predict_lung_cancer(age, smoking, yellow_fingers, anxiety, peer_pressure,
                                     chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                                     coughing, shortness_of_breath, swallowing_difficulty, chest_pain)
        st.success(f"The prediction is: {result}")

if __name__ == "__main__":
    main()


# In[ ]:




