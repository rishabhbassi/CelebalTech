import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# List files in the input directory
for dirname, _, filenames in os.walk('Loan_Prediction'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load datasets
train_data = pd.read_csv('Loan_Prediction/Training Dataset.csv')
test_data = pd.read_csv('Loan_Prediction/Test Dataset.csv')
sample_submission = pd.read_csv('Loan_Prediction/Sample_Submission.csv')

print("Training Data:\n", train_data.head())
print("\nTest Data:\n", test_data.head())
print("\nSample Submission:\n", sample_submission.head())

# Handle missing values in training data
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mean(), inplace=True)

# Handle missing values in testing data
test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True)
test_data['Married'].fillna(test_data['Married'].mode()[0], inplace=True)
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0], inplace=True)
test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(), inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mean(), inplace=True)

# One-hot encode categorical variables
train_data = pd.get_dummies(train_data, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

# Encode target variable
loan_status_le = LabelEncoder()
loan_status_le.fit(train_data['Loan_Status'])
y = loan_status_le.transform(train_data['Loan_Status'])

# Split training data
X = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_val_pred_log = log_reg.predict(X_val)

# Evaluate Logistic Regression model
accuracy_log = accuracy_score(y_val, y_val_pred_log)
classification_rep_log = classification_report(y_val, y_val_pred_log)

# Train Random Forest model
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_val_pred_rf = rf_clf.predict(X_val)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
classification_rep_rf = classification_report(y_val, y_val_pred_rf)

# Choose the better model
if accuracy_log > accuracy_rf:
    final_model = log_reg
else:
    final_model = rf_clf

# Predict on the test set
X_test = test_data.drop('Loan_ID', axis=1)
test_preds = final_model.predict(X_test)
test_preds_decoded = loan_status_le.inverse_transform(test_preds)

# Prepare the submission file
submission = pd.DataFrame({'Loan_ID': test_data['Loan_ID'], 'Loan_Status': test_preds_decoded})
submission.to_csv('Final_Submission.csv', index=False)

print("\nFinal Submission:\n", submission.head())
