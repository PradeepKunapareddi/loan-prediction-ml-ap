from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import pandas as pd

# Initialize app
app = FastAPI(title="Loan Prediction API")

# Load model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# These encodings must match the LabelEncoder mappings used in train_model.py
GENDER_MAP = {"Female": 0, "Male": 1}
MARRIED_MAP = {"No": 0, "Yes": 1}
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}

# Column order the model/scaler were trained on (see notebook.ipynb, section 3)
FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area",
    "TotalIncome", "EMI",
]


class LoanInput(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: float = Field(gt=0)
    Loan_Amount_Term: float = Field(gt=0)
    Credit_History: Literal[0, 1]
    Property_Area: Literal["Rural", "Semiurban", "Urban"]


@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}


@app.post("/predict")
def predict(data: LoanInput):
    try:
        # Feature engineering — must mirror notebook.ipynb exactly
        total_income = data.ApplicantIncome + data.CoapplicantIncome
        emi = data.LoanAmount / data.Loan_Amount_Term

        row = {
            "Gender": GENDER_MAP[data.Gender],
            "Married": MARRIED_MAP[data.Married],
            "Dependents": DEPENDENTS_MAP[data.Dependents],
            "Education": EDUCATION_MAP[data.Education],
            "Self_Employed": SELF_EMPLOYED_MAP[data.Self_Employed],
            "LoanAmount": data.LoanAmount,
            "Loan_Amount_Term": data.Loan_Amount_Term,
            "Credit_History": data.Credit_History,
            "Property_Area": PROPERTY_AREA_MAP[data.Property_Area],
            "TotalIncome": total_income,
            "EMI": emi,
        }

        # Using a DataFrame (not a raw numpy array) keeps column names attached,
        # which avoids sklearn's "X does not have valid feature names" warning
        # and guarantees columns line up with what the scaler was fit on.
        features_df = pd.DataFrame([row], columns=FEATURE_ORDER)

        scaled = scaler.transform(features_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result = "Approved" if prediction == 1 else "Rejected"

        return {
            "prediction": int(prediction),
            "result": result,
            "approval_probability": round(float(probability), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
