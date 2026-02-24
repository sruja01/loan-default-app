import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/Loan_default.csv")

# Drop LoanID (not useful for prediction)
df = df.drop(columns=["LoanID"])

# Drop missing values
df = df.dropna()

# Encode categorical columns
categorical_cols = ["Education", "EmploymentType", "MaritalStatus",
                    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering
df["EMI_Ratio"] = df["LoanAmount"] / df["Income"]

# Features and target
X = df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
       'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
       'LoanPurpose', 'HasCoSigner', 'EMI_Ratio']]
y = df["Default"]
print(X.shape)
print(X.columns)
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Model
# rf = RandomForestClassifier(n_estimisticators=100, random_state=42)
rf = RandomForestClassifier(
    n_estimators=15,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(rf, "model/model.pkl")
joblib.dump(X.columns, 'columns.pkl')
print("Model saved successfully!")

# # Models
# lr = LogisticRegression(max_iter=1000)
# rf = RandomForestClassifier()

# lr.fit(X_train, y_train)
# rf.fit(X_train, y_train)

# # Predictions
# y_pred = rf.predict(X_test)
# y_prob = rf.predict_proba(X_test)[:, 1]

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Save best model
# joblib.dump(rf, "model/model.pkl")

# print("Model saved successfully!")