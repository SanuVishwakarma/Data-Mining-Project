import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# Load the model
model = pickle.load(open(r'C:\VS Code\breast_cancer_ml\breast_cancer_detector.pkl', 'rb'))

st.title("Breast Cancer Prediction Using ML")

# Sidebar navigation
nav = st.sidebar.radio("Navigation", ["Home", "Prediction"])
if nav == "Home":
    st.image("Images/can_symb_hand.jpg")

if nav == "Prediction":
    # Add custom CSS for transparent text
    st.markdown(
        """
        <style>
        .transparent-text input {
            color: rgba(0, 0, 0, 0.5);  /* Adjust the RGBA value for transparency */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # List of placeholders
    placeholders = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
        'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 
        'area error', 'smoothness error', 'compactness error', 'concavity error', 
        'concave points error', 'symmetry error', 'fractal dimension error', 
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 
        'worst fractal dimension'
    ]

    # Initialize a dictionary to store the values from the input boxes
    values = {}

    # Define the number of columns
    num_columns = 5

    # Create the rows and columns of input boxes
    for row in range(0, len(placeholders), num_columns):
        cols = st.columns(num_columns)
        for col, i in zip(cols, range(num_columns)):
            if row + i < len(placeholders):
                with col:
                    value = st.text_input(placeholders[row + i], key=f"value_{row + i + 1}", label_visibility="collapsed", placeholder=placeholders[row + i])
                    values[placeholders[row + i]] = value

    # Define the function to handle the prediction
    def predict():
        # Convert values to the required format (e.g., float) and create input array
        input_values = []
        for key in placeholders:
            try:
                input_values.append(float(values[key]))
            except ValueError:
                st.error(f"Invalid input for {key}")
                return
        
        # Convert list to numpy array and reshape for the model
        input_array = [input_values]
        
        # Call the model's predict function
        prediction = model.predict(input_array)
        if prediction == 0:
            res_val = "Breast Cancer"
        else:
            res_val = "No Breast Cancer"
        st.write("Patient Has", res_val)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        predict()
