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
import numpy as np
import shap # Import the SHAP library
import matplotlib.pyplot as plt # Needed for displaying SHAP plots in Streamlit

# --- Streamlit Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide", # Use a wide layout
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# --- Define Paths ---
model_dir = 'trained_models'
model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
shap_background_path = os.path.join(model_dir, 'X_train_sample_for_shap.joblib') # Path to SHAP background data

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

# --- Load ML Assets with Caching ---
@st.cache_resource # This decorator loads the assets only once across app runs
def load_ml_assets():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        background_data_for_shap = joblib.load(shap_background_path)
        st.sidebar.success("✅ Model, Scaler, & SHAP Data Loaded!") # Visual feedback in sidebar
        return model, scaler, background_data_for_shap
    except FileNotFoundError as e:
        st.error(f"Error loading ML assets: {e}. Make sure the 'trained_models' folder is in the same directory as 'app.py' and contains all necessary files.")
        st.stop() # Stop the app if crucial files aren't found

model, scaler, background_data_for_shap = load_ml_assets()

# --- Cache job options for faster loading ---
@st.cache_data
def get_job_options():
    try:
        original_application_df_path = os.path.join('Dataset', 'application.csv')
        original_application_df = pd.read_csv(original_application_df_path)
        # Add 'Unknown' and sort for consistent order
        return ['Unknown'] + sorted(original_application_df['job'].dropna().unique().tolist())
    except FileNotFoundError:
        st.warning(f"Could not load original 'application.csv' from {original_application_df_path} for job types. Using fallback.")
        # Fallback list if file is not found (for robustness during deployment issues)
        return ['Unknown', 'Sales staff', 'Core staff', 'Managers', 'Laborers', 'Drivers', 'Accountants', 'High skill tech staff', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Secretaries', 'Low-skill Laborers', 'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff']

job_options = get_job_options()

# --- Streamlit App Title and Description ---
st.title("💳 Credit Card Approval Predictor")
st.markdown("""
    Welcome to the **Automated Credit Card Approval System!** Enter an applicant's details below to receive an instant prediction of their credit card approval status,
    along with an explanation of the factors influencing the decision.
""")

# --- Main Content Area ---
st.header("Applicant Information")
st.write("Please fill in the details accurately to get the most precise prediction.")

# Using columns for a nicer layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("🚻 Gender", ["M", "F"], help="Applicant's gender (Male/Female).")
    own_car = st.selectbox("🚗 Own Car", ["Yes", "No"], help="Does the applicant own a car?")
    own_realty = st.selectbox("🏠 Own Realty", ["Yes", "No"], help="Does the applicant own any real estate?")
    num_child = st.number_input("🧒 Number of Children", min_value=0, max_value=20, value=0, help="Total number of children the applicant has.")
    income = st.number_input("💰 Annual Income (USD)", min_value=0.0, value=30000.0, step=1000.0, format="%.2f", help="Applicant's total declared annual income in US Dollars.")
    income_type = st.selectbox("💸 Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"], help="Primary source of the applicant's income.")
    CNT_FAM_MEMBERS = st.number_input("👨‍👩‍👧‍👦 Number of Family Members", min_value=1, max_value=20, value=2, help="Total number of family members.")

with col2:
    education_level = st.selectbox("🎓 Education Level", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"], help="Applicant's highest attained education level.")
    family_status = st.selectbox("❤️ Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], help="Applicant's marital or family status.")
    house_type = st.selectbox("🏡 House / Apartment Type", ["House / apartment", "Rented apartment", "Municipal apartment", "With parents", "Co-op apartment", "Office apartment"], help="Type of housing the applicant resides in.")

    age_in_years = st.number_input("🎂 Age in Years", min_value=18, max_value=100, value=35, help="Applicant's current age in years.")
    years_employed = st.number_input("💼 Years Employed", min_value=0, max_value=50, value=5, help="Total number of years the applicant has been employed. Enter 0 if unemployed.")

    job = st.selectbox("🧑‍💻 Job Type", job_options, help="Applicant's current occupation. 'Unknown' if not specified.")

    max_status = st.selectbox("📈 Max Credit Status (from history)",
                               options=list(range(-2, 6)), # Options from -2 to 5
                               format_func=lambda x: {
                                   -2: '(-2) No Loan',
                                   -1: '(-1) Paid Off',
                                   0: '(0) 1-29 Days Overdue',
                                   1: '(1) 30-59 Days Overdue',
                                   2: '(2) 60-89 Days Overdue',
                                   3: '(3) 90-119 Days Overdue',
                                   4: '(4) 120-149 Days Overdue',
                                   5: '(5) 150+ Days Overdue'
                               }.get(x, str(x)), # Display user-friendly text
                               index=4, # Default to 0 days overdue (index of 0 in options)
                               help="Represents the worst observed credit status from historical records. Higher values indicate more severe delinquency."
                               )

# Add a button to make prediction
if st.button("🚀 Predict Approval", type="primary"):
    st.subheader("📊 Prediction Results:")

    # Use a spinner for better user experience during processing
    with st.spinner("Analyzing applicant data and generating prediction..."):
        # --- 4. Convert User Inputs to Model-Expected Format ---
        own_car_val = "Y" if own_car == "Yes" else "N"
        own_realty_val = "Y" if own_realty == "Yes" else "N"
        birth_day_val = -int(age_in_years * 365.25)
        employment_length_val = -int(years_employed * 365.25)

        user_data = {
            'num_child': num_child, 'income': income, 'birth_day': birth_day_val,
            'employment_length': employment_length_val, 'mobile': 0, 'work_phone': 0,
            'phone': 0, 'email': 0, 'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
            'max_status': max_status, 'gender': gender, 'own_car': own_car_val,
            'own_realty': own_realty_val, 'income_type': income_type,
            'education_level': education_level, 'family_status': family_status,
            'house_type': house_type, 'job': job
        }

        input_df_raw = pd.DataFrame([user_data])

        # --- 5. Preprocessing User Input (MUST match training preprocessing) ---
        # Apply one-hot encoding
        processed_input_df = pd.get_dummies(input_df_raw, columns=ORIGINAL_CATEGORICAL_COLS, drop_first=True)

        # Align columns to match the exact order and presence of training data (FINAL_TRAINED_FEATURES)
        final_input_for_prediction = pd.DataFrame(columns=FINAL_TRAINED_FEATURES)
        for col in FINAL_TRAINED_FEATURES:
            if col in processed_input_df.columns:
                final_input_for_prediction[col] = processed_input_df[col]
            else:
                final_input_for_prediction[col] = 0

        # Ensure all numerical columns are of numeric type for scaling
        all_numerical_cols_in_final_features = [
            'num_child', 'income', 'birth_day', 'employment_length', 'mobile',
            'work_phone', 'phone', 'email', 'CNT_FAM_MEMBERS', 'max_status'
        ]
        for col in all_numerical_cols_in_final_features:
            if col in final_input_for_prediction.columns:
                final_input_for_prediction[col] = pd.to_numeric(final_input_for_prediction[col], errors='coerce')

        final_input_for_prediction = final_input_for_prediction.fillna(0)

        # Apply Feature Scaling
        final_input_for_prediction[NUMERICAL_COLS_TO_SCALE] = scaler.transform(final_input_for_prediction[NUMERICAL_COLS_TO_SCALE])

        # --- 6. Make Prediction ---
        prediction = model.predict(final_input_for_prediction)
        prediction_proba = model.predict_proba(final_input_for_prediction)[:, 1] # Probability of approval (class 1)

    # --- 7. Display Prediction ---
    if prediction[0] == 1:
        st.success(f"**Prediction: Approved!** ✅")
        st.balloons() # Add balloons for success!
    else:
        st.error(f"**Prediction: Rejected.** ❌")

    st.markdown(f"**Probability of Approval:** `{prediction_proba[0]:.2f}`")

    st.info("💡 *Remember: This prediction is based on the trained model. Real-world credit decisions involve additional factors and human oversight.*")

    # --- 8. SHAP Interpretability ---
    st.subheader("🧐 Why this prediction? (Feature Contributions)")
    st.markdown("The SHAP force plot below illustrates how each feature contributes to the prediction. Features in **red** push the prediction higher (towards Approval), while features in **blue** push it lower (towards Rejection).")

    try:
        # Initialize SHAP explainer with the loaded background data
        explainer = shap.KernelExplainer(model.predict_proba, background_data_for_shap)

        # Calculate SHAP values for the current input
        # Ensure the input is passed as a DataFrame row for SHAP.
        shap_values = explainer.shap_values(final_input_for_prediction.iloc[0])

        # Get SHAP values for the 'Approved' class (class 1)
        shap_values_class_1 = shap_values[1]

        # Get the expected value for class 1 (average model output)
        expected_value_class_1 = explainer.expected_value[1]

        shap.initjs() # Initialize JavaScript for SHAP plots (important for rendering)

        # Create the force plot HTML
        shap_html = shap.force_plot(
            expected_value_class_1,
            shap_values_class_1,
            final_input_for_prediction.iloc[0], # The feature values for the single input
            feature_names=FINAL_TRAINED_FEATURES # Use your defined list of feature names
        )

        # Display in an expander for cleaner UI, allowing users to show/hide it
        with st.expander("✨ Click to view detailed Feature Contributions (SHAP Force Plot)", expanded=True):
            # Using st.components.v1.html to embed the plot, ensure scrolling is enabled
            st.components.v1.html(shap.get_html(shap_html), width=900, height=350, scrolling=True)
            st.markdown("""
            **How to read the SHAP Force Plot:**
            - **Base Value (f(x) on the plot):** This is the average model output (probability of approval) if no features were considered.
            - **Red values** on the right push the prediction **higher** (towards Approval).
            - **Blue values** on the left push the prediction **lower** (towards Rejection).
            - Each feature's position indicates its impact on pushing the prediction from the base value to the final output value.
            """)
    except Exception as e:
        st.error(f"Could not generate SHAP explanation. Error: {e}")
        st.warning("SHAP explanations might not work perfectly with all model types or specific input values.")


# --- Professional Footer ---
st.markdown("---")
st.markdown("A Machine Learning Project by Jahnavi Israni for Internship Use.")
st.markdown("Built with Python, Scikit-learn, and Streamlit.")