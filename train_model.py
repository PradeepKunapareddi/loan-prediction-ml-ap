"""
Loan Prediction - Model Training Script

Loads the raw loan application data, cleans it, engineers features,
trains a Random Forest classifier, evaluates it, and saves the trained
model and scaler to disk so they can be served by app.py (FastAPI).

Run with:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv("train.csv")
    print(df.head())
    print()
    df.info()
    print()
    print(df.describe())

    # ------------------------------------------------------------------
    # 2. Handle missing values
    # ------------------------------------------------------------------
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Fill categorical columns with the mode, numeric columns with mean/constant.
    
   
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
    df["Credit_History"] = df["Credit_History"].fillna(1)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # ------------------------------------------------------------------
    # 3. Encode the target column
    # ------------------------------------------------------------------
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    print("\nTarget distribution:")
    print(df["Loan_Status"].value_counts()) 

    # ------------------------------------------------------------------
    # 4. Feature engineering
    #    TotalIncome = ApplicantIncome + CoapplicantIncome
    #    EMI = LoanAmount / Loan_Amount_Term (proxy for monthly repayment burden)
    #    Then drop the raw income columns and the identifier column
    #    (Loan_ID is unique per row and carries no predictive signal). 
    # ------------------------------------------------------------------
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"] 

    X = df.drop(columns=["Loan_ID", "ApplicantIncome", "CoapplicantIncome", "Loan_Status"])
    y = df["Loan_Status"] 

    # ------------------------------------------------------------------
    # 5. Encode categorical features
    # ------------------------------------------------------------------
    categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]
    encoders = {}

    for col in categorical_cols: 
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(col, "->", dict(zip(le.classes_, le.transform(le.classes_))))

    # ------------------------------------------------------------------
    # 6. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 7. Feature scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 8. Train the model
    #    The dataset is imbalanced (far more "Approved" than "Rejected" loans),
    #    so class_weight gives more weight to the minority class (0 = Rejected).
    # ------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        class_weight={0: 4, 1: 1},
        random_state=42,
    )
    model.fit(X_train_scaled, y_train) 

    # ------------------------------------------------------------------
    # 9. Evaluate 
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled) 

    print("\nAccuracy:", accuracy_score(y_test, y_pred)) 
    print()
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------
    # 10. Save the model and scaler
    #     These are the exact artifacts app.py loads at startup. The column
    #     order here (Gender, Married, Dependents, Education, Self_Employed,
    #     LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
    #     TotalIncome, EMI) must match the order app.py builds its feature
    #     vector in.
    # ------------------------------------------------------------------
    joblib.dump(model, "loan_model.pkl") 
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved loan_model.pkl and scaler.pkl")


if __name__ == "__main__": 
    main()
