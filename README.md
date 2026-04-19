Loan Prediction ML API

📌 Project Overview

This project predicts whether a loan will be Approved or Rejected using Machine Learning.

It is built using:

 * Python
 * Scikit-learn
 * FastAPI

⚙️ Features

 * Data preprocessing & cleaning
 * Feature engineering (TotalIncome, EMI)
 * Handling class imbalance
 * Model training using RandomForest
 * REST API using FastAPI

📊 Input Features

 * Gender
 * Married
 * Dependents
 * Education
 * Self_Employed
 * LoanAmount
 * Loan_Amount_Term
 * Credit_History
 * Property_Area

How to Run

1. Install dependencies
pip install -r requirements.txt

3. Run API
uvicorn app:app --reload

5. Open Swagger UI
http://127.0.0.1:8000/docs


🧠 Model Details

* Algorithm: Random Forest
* Class imbalance handled using class weights
* Feature engineering used for better prediction


📌 Example Output
{
  "prediction": 1,
  "result": "Approved"
}

   
 
