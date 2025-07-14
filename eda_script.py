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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import joblib # Added for saving SHAP background data
import os # Added for path operations

# Define the path to your dataset folder relative to where your script is saved
dataset_path = 'Dataset/'

# Load the application data
try:
    application_df = pd.read_csv(dataset_path + 'application.csv')
    print("application.csv loaded successfully!")
except FileNotFoundError:
    print(f"Error: application.csv not found at {dataset_path}application.csv. "
          "Please check your file path and ensure the 'Dataset' folder is in the same directory as your script.")

# Load the credit record data
try:
    credit_record_df = pd.read_csv(dataset_path + 'credit_record.csv')
    print("credit_record.csv loaded successfully!")
except FileNotFoundError:
    print(f"Error: credit_record.csv not found at {dataset_path}credit_record.csv. "
          "Please check your file path and ensure the 'Dataset' folder is in the same directory as your script.")

# --- Initial Inspection ---
print("\n--- Initial Inspection of application_df ---")
print("Shape:", application_df.shape)
print("\nFirst 5 rows:")
print(application_df.head())
print("\nColumn Info:")
application_df.info()
print("\nDescriptive Statistics:")
print(application_df.describe())

print("\n--- Initial Inspection of credit_record_df ---")
print("Shape:", credit_record_df.shape)
print("\nFirst 5 rows:")
print(credit_record_df.head())
print("\nColumn Info:")
credit_record_df.info()
print("\nDescriptive Statistics:")
print(credit_record_df.describe())

# --- Identify Missing Values ---
print("\n--- Missing Values in application_df ---")
print(application_df.isnull().sum())
print("\nPercentage of Missing Values in application_df:")
print(application_df.isnull().sum() / len(application_df) * 100)

print("\n--- Missing Values in credit_record_df ---")
print(credit_record_df.isnull().sum())
print("\nPercentage of Missing Values in credit_record_df:")
print(credit_record_df.isnull().sum() / len(credit_record_df) * 100)

# --- Explore Categorical Features in application_df ---
print("\n--- Exploring Categorical Features in application_df ---")
categorical_cols_app = application_df.select_dtypes(include='object').columns

for col in categorical_cols_app:
    print(f"\nValue Counts for '{col}':")
    print(application_df[col].value_counts())
    print(f"\nPercentage Value Counts for '{col}':")
    print(application_df[col].value_counts(normalize=True) * 100)

    if col != 'job':
        plt.figure(figsize=(10, 6))
        sns.countplot(data=application_df, y=col, order=application_df[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()
        print(f"Plot saved for {col}: {col}_distribution.png")
    else:
        plt.figure(figsize=(12, 8))
        sns.countplot(data=application_df, y=col, order=application_df[col].value_counts(dropna=False).index, palette='viridis')
        plt.title(f'Distribution of {col} (including Missing)')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f'{col}_distribution_with_nan.png')
        plt.close()
        print(f"Plot saved for {col} (including Missing): {col}_distribution_with_nan.png")

# --- Explore Categorical Features in credit_record_df ---
print("\n--- Exploring Categorical Features in credit_record_df ---")
categorical_cols_credit = credit_record_df.select_dtypes(include='object').columns

for col in categorical_cols_credit:
    print(f"\nValue Counts for '{col}':")
    print(credit_record_df[col].value_counts())
    print(f"\nPercentage Value Counts for '{col}':")
    print(credit_record_df[col].value_counts(normalize=True) * 100)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=credit_record_df, y=col, order=credit_record_df[col].value_counts().index, palette='crest')
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_distribution_credit.png')
    plt.close()
    print(f"Plot saved for {col}: {col}_distribution_credit.png")

# --- Explore Numerical Features in application_df ---
print("\n--- Exploring Numerical Features in application_df ---")
numerical_cols_app = application_df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols_app:
    if col in ['Unnamed: 0', 'id']:
        continue
    print(f"\nDescriptive Statistics for '{col}':")
    print(application_df[col].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=application_df, x=col, kde=True, bins=50, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{col}_distribution_hist.png')
    plt.close()
    print(f"Histogram saved for {col}: {col}_distribution_hist.png")

    plt.figure(figsize=(10, 2))
    sns.boxplot(data=application_df, x=col, palette='viridis')
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_boxplot.png')
    plt.close()
    print(f"Box plot saved for {col}: {col}_boxplot.png")

# --- Explore Numerical Features in credit_record_df ---
print("\n--- Exploring Numerical Features in credit_record_df ---")
numerical_cols_credit = credit_record_df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols_credit:
    if col in ['Unnamed: 0', 'id']:
        continue
    print(f"\nDescriptive Statistics for '{col}':")
    print(credit_record_df[col].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=credit_record_df, x=col, kde=True, bins=50, palette='crest')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{col}_distribution_credit_hist.png')
    plt.close()
    print(f"Histogram saved for {col}: {col}_distribution_credit_hist.png")

    plt.figure(figsize=(10, 2))
    sns.boxplot(data=credit_record_df, x=col, palette='crest')
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_boxplot_credit.png')
    plt.close()
    print(f"Box plot saved for {col}: {col}_boxplot_credit.png")

# --- Correlation Matrix for application_df (Numerical Features) ---
print("\n--- Correlation Matrix for application_df Numerical Features ---")
cols_for_corr_app = [col for col in numerical_cols_app if col not in ['Unnamed: 0', 'id']]
correlation_matrix_app = application_df[cols_for_corr_app].corr()
print(correlation_matrix_app)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_app, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (application_df)')
plt.tight_layout()
plt.savefig('application_df_correlation_matrix.png')
plt.close()
print("Correlation matrix heatmap saved: application_df_correlation_matrix.png")

# --- Correlation Matrix for credit_record_df (Numerical Features) ---
print("\n--- Correlation Matrix for credit_record_df Numerical Features ---")
cols_for_corr_credit = [col for col in numerical_cols_credit if col not in ['Unnamed: 0', 'id']]
correlation_matrix_credit = credit_record_df[cols_for_corr_credit].corr()
print(correlation_matrix_credit)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_credit, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (credit_record_df)')
plt.tight_layout()
plt.savefig('credit_record_df_correlation_matrix.png')
plt.close()
print("Correlation matrix heatmap saved: credit_record_df_correlation_matrix.png")

# --- Understanding the Target Variable and Data Linkage ---
print("\n--- Processing credit_record_df for Target Variable ---")
credit_record_df['status_numeric'] = credit_record_df['status'].replace({'C': -1, 'X': -2}).astype(int)

credit_grouped = credit_record_df.groupby('id')['status_numeric'].max().reset_index()
credit_grouped.rename(columns={'status_numeric': 'max_status'}, inplace=True)

credit_grouped['target'] = credit_grouped['max_status'].apply(lambda x: 0 if x >= 2 else 1)

print("\nSample of credit_grouped with new 'target' variable:")
print(credit_grouped.head())
print("\nDistribution of 'target' variable (1=Approved, 0=Rejected) from credit_record:")
print(credit_grouped['target'].value_counts())
print("\nPercentage Distribution of 'target' variable:")
print(credit_grouped['target'].value_counts(normalize=True) * 100)

# Merge application_df with the newly created credit_grouped DataFrame
print("\n--- Merging application_df with credit_grouped ---")
merged_df = pd.merge(application_df, credit_grouped, on='id', how='inner')

print("\nShape of the merged DataFrame:", merged_df.shape)
print("\nFirst 5 rows of merged DataFrame (showing 'id', 'job', and 'target'):")
print(merged_df[['id', 'job', 'gender', 'income', 'target']].head())

print("\n--- Final Target Variable Distribution in Merged DataFrame ---")
print(merged_df['target'].value_counts())
print("\nFinal Percentage Distribution of 'target' variable (1=Approved, 0=Rejected):")
print(merged_df['target'].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 6))
sns.countplot(data=merged_df, x='target', palette='coolwarm')
plt.title('Final Distribution of Target Variable (0=Rejected, 1=Approved)')
plt.xlabel('Credit Approval Status')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Rejected', 'Approved'])
plt.tight_layout()
plt.savefig('final_target_distribution.png')
plt.close()
print("Final target distribution plot saved: final_target_distribution.png")

print("\n--- EDA Step 1 Completed! ---")

# --- Step 2: Data Preprocessing and Feature Engineering ---
print("\n--- Starting Step 2: Data Preprocessing and Feature Engineering ---")

# 1. Handling Missing Values
print("\n--- Handling Missing Values ---")
print(f"Missing values in 'job' before imputation: {merged_df['job'].isnull().sum()}")
merged_df['job'].fillna('Unknown', inplace=True)
print(f"Missing values in 'job' after imputation: {merged_df['job'].isnull().sum()}")

# 2. Drop unnecessary columns
print("\n--- Dropping 'Unnamed: 0' and 'id' columns ---")
initial_columns = merged_df.columns.tolist()
merged_df.drop(columns=['Unnamed: 0', 'id'], inplace=True)
print(f"Columns before dropping: {initial_columns}")
print(f"Columns after dropping: {merged_df.columns.tolist()}")

# Verify the changes
print("\n--- Verifying DataFrame after initial preprocessing ---")
print("New Shape:", merged_df.shape)
print("\nCheck for any remaining missing values:")
print(merged_df.isnull().sum().sum())

print("\nFinished handling missing values and dropping ID columns. Proceeding to categorical processing.")

# --- 3. Processing Categorical Features ---
print("\n--- Processing Categorical Features (One-Hot Encoding) ---")
categorical_cols_to_encode = merged_df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns to be encoded: {categorical_cols_to_encode}")

merged_df_encoded = pd.get_dummies(merged_df, columns=categorical_cols_to_encode, drop_first=True)
print("\nShape after One-Hot Encoding:", merged_df_encoded.shape)
print("\nFirst 5 rows after One-Hot Encoding (sample of new columns):")
print(merged_df_encoded.head())
print("\nFinished processing categorical features. Proceeding to feature scaling.")

# --- 4. Feature Scaling ---
print("\n--- Starting Feature Scaling ---")

X = merged_df_encoded.drop('target', axis=1)
y = merged_df_encoded['target']

numerical_cols_to_scale = ['num_child', 'income', 'birth_day', 'employment_length', 'CNT_FAM_MEMBERS']

scaler = StandardScaler()
print(f"Applying StandardScaler to columns: {numerical_cols_to_scale}")
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

print("\nShape of X (features) after scaling:", X.shape)
print("\nFirst 5 rows of X (features) after scaling (sample of scaled columns):")
print(X[numerical_cols_to_scale].head())
print("\nFinished Feature Scaling.")
print("\n--- Step 2: Data Preprocessing and Feature Engineering Completed! ---")

# --- Step 3: Handling Imbalanced Data ---
print("\n--- Starting Step 3: Handling Imbalanced Data ---")

print("\nOriginal X shape:", X.shape)
print("Original y shape:", y.shape)
print("Original target distribution (before splitting):")
print(y.value_counts(normalize=True) * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print("\nTraining target distribution (before SMOTE):")
print(y_train.value_counts(normalize=True) * 100)

print("\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
print(f"Shape of y_train after SMOTE: {y_train_resampled.shape}")
print("\nTraining target distribution (after SMOTE):")
print(y_train_resampled.value_counts(normalize=True) * 100)

print("\n--- Step 3: Handling Imbalanced Data Completed! ---")

# --- Step 4: Model Building and Training (Logistic Regression) ---
print("\n--- Starting Step 4: Model Building and Training (Logistic Regression) ---")

log_reg_model = LogisticRegression(solver='liblinear', random_state=42, C=0.1, max_iter=1000)

print("\nTraining Logistic Regression model...")
log_reg_model.fit(X_train_resampled, y_train_resampled)

print("\nModel training completed.")

print("\n--- Evaluating Model on Test Set ---")
y_pred = log_reg_model.predict(X_test)
y_pred_proba = log_reg_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score (Area Under the Receiver Operating Characteristic Curve):")
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("True Negatives (TN):", cm[0, 0])
print("False Positives (FP):", cm[0, 1])
print("False Negatives (FN):", cm[1, 0])
print("True Positives (TP):", cm[1, 1])

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()
print("ROC Curve plot saved: roc_curve.png")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
plt.close()
print("Precision-Recall Curve plot saved: precision_recall_curve.png")

print("\n--- Step 4: Model Building and Training Completed! ---")

# --- Step 5: Hyperparameter Optimization (GridSearchCV) ---
print("\n--- Starting Step 5: Hyperparameter Optimization (GridSearchCV) ---")

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

log_reg_grid = LogisticRegression(random_state=42, max_iter=2000)

grid_search = GridSearchCV(estimator=log_reg_grid, param_grid=param_grid,
                           cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

print("\nPerforming Grid Search Cross-Validation for Hyperparameter Optimization...")
grid_search.fit(X_train_resampled, y_train_resampled)

print("\nGrid Search completed.")

best_params = grid_search.best_params_
best_roc_auc = grid_search.best_score_
best_model = grid_search.best_estimator_

print(f"\nBest Hyperparameters found: {best_params}")
print(f"Best ROC AUC Score from cross-validation: {best_roc_auc:.4f}")

print("\n--- Evaluating Best Model on Test Set (after Hyperparameter Optimization) ---")
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

print("\nROC AUC Score for Best Model on Test Set:")
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)
print(f"ROC AUC: {roc_auc_best:.4f}")

print("\nConfusion Matrix for Best Model:")
cm_best = confusion_matrix(y_test, y_pred_best)
print(cm_best)
print("True Negatives (TN):", cm_best[0, 0])
print("False Positives (FP):", cm_best[0, 1])
print("False Negatives (FN):", cm_best[1, 0])
print("True Positives (TP):", cm_best[1, 1])

fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_proba_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_best)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Best Model)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_best_model.png')
plt.close()
print("ROC Curve plot for best model saved: roc_curve_best_model.png")

precision_best, recall_best, _ = precision_recall_curve(y_test, y_pred_proba_best)
pr_auc_best = auc(recall_best, precision_best)
plt.figure(figsize=(8, 6))
plt.plot(recall_best, precision_best, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc_best)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Best Model)')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('precision_recall_curve_best_model.png')
plt.close()
print("Precision-Recall Curve plot for best model saved: precision_recall_curve_best_model.png")

print("\n--- Step 5: Hyperparameter Optimization Completed! ---")
print("\nYour core ML project build is now complete. You have a trained and optimized model!")

# --- Step 6: Model Persistence (Saving Model and Scaler) ---
print("\n--- Starting Step 6: Model Persistence (Saving Model and Scaler) ---")

model_dir = 'trained_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

model_filename = os.path.join(model_dir, 'logistic_regression_model.joblib')
joblib.dump(best_model, model_filename)
print(f"Trained Logistic Regression model saved to: {model_filename}")

scaler_filename = os.path.join(model_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_filename)
print(f"StandardScaler saved to: {scaler_filename}")

# --- NEW: Save X_train_sample for SHAP background ---
# This sample is used by SHAP's KernelExplainer as its background dataset
# for more accurate and representative explanations.
X_train_sample_for_shap = X_train_resampled.sample(n=100, random_state=42) # Adjust n as needed
joblib.dump(X_train_sample_for_shap, os.path.join(model_dir, 'X_train_sample_for_shap.joblib'))
print("X_train_sample for SHAP saved.")


print("\n--- Step 6: Model Persistence Completed! ---")
print("\nYour model and scaler are now saved and ready for deployment!")

print("\n--- Final X columns for app.py (copy this list carefully!) ---")
print(X.columns.tolist())
