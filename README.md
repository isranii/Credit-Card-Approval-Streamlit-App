# 💳 Automated Credit Card Approval Predictor

## Project Overview

Developed an optimized machine learning model for automated credit card approval prediction, deployed as an interactive web application. The project involved a full ML pipeline, including robust data preprocessing, handling class imbalance with SMOTE, and hyperparameter optimization for a Logistic Regression model. The solution provides real-time, data-driven approval forecasts through a user-friendly Streamlit interface.

## Features

* **Data Ingestion & EDA:** Comprehensive exploration of applicant and credit history data.
* **Robust Preprocessing:** Handles missing values, performs categorical encoding (One-Hot Encoding), and scales numerical features.
* **Imbalanced Data Handling:** Utilizes SMOTE to effectively manage class imbalance in the target variable.
* **Optimized ML Model:** Implements Logistic Regression with hyperparameters tuned using GridSearchCV for high performance.
* **Model Persistence:** Saves trained model and scaler for efficient deployment.
* **Interactive Web Application:** User-friendly interface built with Streamlit for real-time predictions.

## Tech Stack

* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Logistic Regression, StandardScaler, GridSearchCV)
* **Imbalanced Learning:** Imbalanced-learn (SMOTE)
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit
* **Version Control:** Git, GitHub

## How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/isranii/Credit-Card-Approval-Streamlit-App.git](https://github.com/isranii/Credit-Card-Approval-Streamlit-App.git)
    cd Credit-Card-Approval-Streamlit-App
    ```
    *(Replace `https://github.com/isranii/Credit-Card-Approval-Streamlit-App.git` with your actual repository URL)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the EDA Script (Optional, for re-generating model and plots):**
    This script performs all data preprocessing, model training, and saves the trained model and scaler.
    ```bash
    python eda_script.py
    ```
    *(Note: This might take some time as it includes model training and hyperparameter optimization.)*

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    Your application will open in your default web browser.

## Project Structure

* `app.py`: The main Streamlit web application script.
* `eda_script.py`: Contains all the code for data loading, EDA, preprocessing, model training, and saving.
* `requirements.txt`: Lists all Python dependencies required for the project.
* `Dataset/`: Contains the raw `application.csv` and `credit_record.csv` datasets.
* `trained_models/`: Stores the saved `logistic_regression_model.joblib` and `scaler.joblib` files.
* `.png` image files: Various visualizations and model evaluation plots generated during EDA.
* `main.ipynb` (Optional): Jupyter Notebook version of the analysis, if provided in the original repo.
* `Team_12_Project_Report.pdf` (Optional): Project report from the original repo.

## Live Application

You can access the live deployed application here:
[Link to your Streamlit Cloud App]
*(After deployment, Streamlit Cloud will give you a public URL. Paste that here!)*

## Future Enhancements (Ideas for Continued Improvement)

* **Model Interpretability:** Integrate SHAP (SHapley Additive exPlanations) to provide insights into individual prediction outcomes within the Streamlit app.
* **Advanced Models:** Experiment with other machine learning algorithms (e.g., Random Forest, XGBoost) and compare their performance.
* **Dynamic Threshold Adjustment:** Allow users to adjust the prediction probability threshold to explore the trade-offs between precision and recall.
* **More Features:** Incorporate additional relevant features, potentially from external data sources (e.g., credit scores).
* **Time-Series Analysis:** Explore more sophisticated processing of `credit_record.csv` to capture temporal patterns more effectively.

## Credits / Contact
* **JAHNAVI ISRANI**
* **GitHub:** [https://github.com/isranii](https://github.com/isranii)
* **LinkedIn:** [https://www.linkedin.com/in/jahnaviisrani/](https://www.linkedin.com/in/jahnaviisrani/)
---