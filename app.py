import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
medical_dataset = pd.read_csv('insurance (1).csv')

# Preprocessing
medical_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Splitting the features and target
X = medical_dataset.drop(columns='charges', axis=1)
Y = medical_dataset['charges']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Training the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Streamlit App

# Set the title of the app
st.title('Insurance Charges Prediction')

# Input fields for user data
age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])

# Convert categorical values to numeric
sex = 0 if sex == 'Male' else 1
smoker = 0 if smoker == 'Yes' else 1
region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}
region = region_mapping[region]

# Predict button
if st.button('Predict'):
    # Convert the inputs into an array
    input_data = (age, sex, bmi, children, smoker, region)
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1, -1)

    # Perform the prediction
    prediction = regressor.predict(input_data_reshaped)

    # Show the prediction result
    st.subheader(f"The predicted insurance charges are: ${prediction[0]:.2f}")

# Show footer information
st.write("Note: This prediction is based on a Linear Regression model trained on the given insurance dataset.")
