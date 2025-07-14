#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Jahnavi Israni
#
# Created:     14-07-2025
# Copyright:   (c) Jahnavi Israni 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np # Added for potential numerical operations like rounding

# --- 1. Load the Trained Model and Scaler ---
# Define the directory where your model and scaler are saved
model_dir = 'trained_models'
model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')

# Check if model files exist before trying to load them
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Model and Scaler loaded successfully!")
else:
    st.error("Error: Model or Scaler files not found. Make sure 'trained_models' folder is in the same directory as 'app.py'.")
    st.stop() # Stop the app if files aren't found

# --- CRITICAL: Define the exact list of columns your model was trained on ---
# This list comes directly from the output of X.columns.tolist() in your eda_script.py
FINAL_TRAINED_FEATURES = ['num_child', 'income', 'birth_day', 'employment_length', 'mobile', 'work_phone', 'phone', 'email', 'CNT_FAM_MEMBERS', 'max_status', 'gender_M', 'own_car_Y', 'own_realty_Y', 'income_type_Pensioner', 'income_type_State servant', 'income_type_Student', 'income_type_Working', 'education_level_Higher education', 'education_level_Incomplete higher', 'education_level_Lower secondary', 'education_level_Secondary / secondary special', 'family_status_Married', 'family_status_Separated', 'family_status_Single / not married', 'family_status_Widow', 'house_type_House / apartment', 'house_type_Municipal apartment', 'house_type_Office apartment', 'house_type_Rented apartment', 'house_type_With parents', 'job_Cleaning staff', 'job_Cooking staff', 'job_Core staff', 'job_Drivers', 'job_HR staff', 'job_High skill tech staff', 'job_IT staff', 'job_Laborers', 'job_Low-skill Laborers', 'job_Managers', 'job_Medicine staff', 'job_Private service staff', 'job_Realty agents', 'job_Sales staff', 'job_Secretaries', 'job_Security staff', 'job_Unknown', 'job_Waiters/barmen staff']

# Define the numerical columns that were scaled in eda_script.py
NUMERICAL_COLS_TO_SCALE = ['num_child', 'income', 'birth_day', 'employment_length', 'CNT_FAM_MEMBERS']

# Define the original categorical columns that were one-hot encoded
ORIGINAL_CATEGORICAL_COLS = [
    'gender', 'own_car', 'own_realty', 'income_type', 'education_level',
    'family_status', 'house_type', 'job'
]

# --- 2. Streamlit App Title and Description ---
st.title("💳 Credit Card Approval Predictor")
st.write("Enter applicant details below to predict credit card approval status.")

# --- 3. Input Fields for Applicant Data ---
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    own_car = st.selectbox("Own Car", ["Yes", "No"]) # Changed to Yes/No
    own_realty = st.selectbox("Own Realty", ["Yes", "No"]) # Changed to Yes/No
    num_child = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    income = st.number_input("Annual Income (USD)", min_value=0.0, value=20000.0, step=1000.0)
    income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    CNT_FAM_MEMBERS = st.number_input("Number of Family Members", min_value=1, max_value=20, value=2) # Re-added this input

with col2:
    education_level = st.selectbox("Education Level", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])
    family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    house_type = st.selectbox("House / Apartment Type", ["House / apartment", "Rented apartment", "Municipal apartment", "With parents", "Co-op apartment", "Office apartment"])

    # Changed to Age in Years
    age_in_years = st.number_input("Age in Years", min_value=18, max_value=100, value=30)
    # Changed to Years Employed
    years_employed = st.number_input("Years Employed", min_value=0, max_value=50, value=5)

    try:
        # Load unique job types from the original CSV for dropdown, include 'Unknown'
        original_application_df_path = os.path.join('Dataset', 'application.csv')
        original_application_df = pd.read_csv(original_application_df_path)
        job_options = ['Unknown'] + sorted(original_application_df['job'].dropna().unique().tolist())
    except FileNotFoundError:
        st.error(f"Could not load original 'application.csv' from {original_application_df_path} to get job types. Using fallback.")
        job_options = ['Unknown', 'Sales staff', 'Core staff', 'Managers', 'Laborers', 'Drivers', 'Accountants', 'High skill tech staff', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Secretaries', 'Low-skill Laborers', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff'] # Fallback
    job = st.selectbox("Job Type", job_options)

    # max_status input remains as it is a derived feature from credit history
    max_status = st.selectbox("Max Credit Status (from history, -2=No Loan, -1=Paid Off, 0=1-29 days overdue, 5=150+ days overdue)", list(range(-2, 6)), index=2)

# Add a button to make prediction
if st.button("Predict Approval"):
    st.subheader("Prediction Result:")

    # --- 4. Convert User Inputs to Model-Expected Format ---
    # Convert user-friendly inputs back to model-expected format
    # 'Y'/'N' for own_car/own_realty
    own_car_val = "Y" if own_car == "Yes" else "N"
    own_realty_val = "Y" if own_realty == "Yes" else "N"

    # Convert years to negative days for birth_day and employment_length
    # Using -365.25 for average days in a year to account for leap years
    birth_day_val = -int(age_in_years * 365.25)
    employment_length_val = -int(years_employed * 365.25)

    user_data = {
        'num_child': num_child,
        'income': income,
        'birth_day': birth_day_val,
        'employment_length': employment_length_val,
        'mobile': 0, # Assuming these are flags not explicitly input by user, default to 0
        'work_phone': 0,
        'phone': 0,
        'email': 0,
        'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS, # Now taking input from UI
        'max_status': max_status,
        'gender': gender, # M or F, stays as is for OHE
        'own_car': own_car_val, # Converted to Y/N
        'own_realty': own_realty_val, # Converted to Y/N
        'income_type': income_type,
        'education_level': education_level,
        'family_status': family_status,
        'house_type': house_type,
        'job': job
    }

    # Create a DataFrame from the single row of user input
    input_df_raw = pd.DataFrame([user_data])

    # --- 5. Preprocessing User Input (MUST match training preprocessing) ---
    st.write("Applying preprocessing steps to your input...")

    # A. One-Hot Encoding for categorical features
    # Use pd.get_dummies to encode the categorical features.
    # The `drop_first=True` must match your training.
    # CRITICAL FIX: Changed `pd.dummies` to `pd.get_dummies`
    processed_input_df = pd.get_dummies(input_df_raw, columns=ORIGINAL_CATEGORICAL_COLS, drop_first=True)

    # B. Align columns to match the exact order and presence of training data (FINAL_TRAINED_FEATURES)
    # Create an empty DataFrame with the exact columns and order of the training data
    final_input_for_prediction = pd.DataFrame(columns=FINAL_TRAINED_FEATURES)

    # Copy the values from the processed_input_df to this new DataFrame
    for col in FINAL_TRAINED_FEATURES:
        if col in processed_input_df.columns:
            final_input_for_prediction[col] = processed_input_df[col]
        else:
            final_input_for_prediction[col] = 0 # Fill missing one-hot encoded columns (categories not in this input) with 0

    # Ensure all numerical columns are of numeric type for scaling
    # Include all numerical features that were part of X, not just the ones scaled.
    all_numerical_cols_in_final_features = [
        'num_child', 'income', 'birth_day', 'employment_length', 'mobile',
        'work_phone', 'phone', 'email', 'CNT_FAM_MEMBERS', 'max_status'
    ]
    for col in all_numerical_cols_in_final_features:
        if col in final_input_for_prediction.columns:
            final_input_for_prediction[col] = pd.to_numeric(final_input_for_prediction[col], errors='coerce')

    # Fill any remaining NaNs (e.g., from categories not present in a given input) with 0
    final_input_for_prediction = final_input_for_prediction.fillna(0)

    # C. Apply Feature Scaling to numerical columns
    # Use the loaded `scaler` object.
    final_input_for_prediction[NUMERICAL_COLS_TO_SCALE] = scaler.transform(final_input_for_prediction[NUMERICAL_COLS_TO_SCALE])

    # --- 6. Make Prediction ---
    prediction = model.predict(final_input_for_prediction)
    prediction_proba = model.predict_proba(final_input_for_prediction)[:, 1] # Probability of approval (class 1)

    # --- 7. Display Prediction ---
    if prediction[0] == 1:
        st.success(f"Prediction: **Approved!** ✅")
    else:
        st.error(f"Prediction: **Rejected.** ❌")

    st.write(f"Probability of Approval: **{prediction_proba[0]:.2f}**")

    st.info("Remember: This is a prediction based on the model. Real-world decisions involve more factors.")

# --- Important Notes for Running ---
# You won't run this script directly in PyScripter.
# 1. Save this file (Ctrl+S) in PyScripter.
# 2. Go to your Command Prompt (where you ran 'streamlit run app.py' before).
# 3. If it's still running, press Ctrl+C to stop it.
# 4. Then, run 'streamlit run app.py' again to see the changes.