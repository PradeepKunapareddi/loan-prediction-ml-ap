from fastapi import FastAPI 
from pydantic import BaseModel 
import joblib 
import numpy as np 

#Initialize app
app = FastAPI(title="Loan prediction API")

# Load model and scaler 
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: int

@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}
    
@app.post("/predict")
def predict(data:LoanInput):
    try: 
        features = np.array([[
            data.Gender,
            data.Married,
            data.Dependents,
            data.Education,
            data.Self_Employed,
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History,
            data.Property_Area

        ]])

        #Scale
        features = scaler.transform(features)

        #Predict
        prediction = model.predict(features)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        return {
            "prediction": int(prediction),
            "result"    : result 
        }
    except Exception as e:
        return {"error": str(e)}
    












