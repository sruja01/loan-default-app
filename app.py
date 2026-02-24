from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return "Model file not found. Please train the model first."

    # Get form values
    Age = float(request.form['Age'])
    Income = float(request.form['Income'])
    LoanAmount = float(request.form['LoanAmount'])
    CreditScore = float(request.form['CreditScore'])
    MonthsEmployed = float(request.form['MonthsEmployed'])
    NumCreditLines = float(request.form['NumCreditLines'])
    InterestRate = float(request.form['InterestRate'])
    LoanTerm = float(request.form['LoanTerm'])
    DTIRatio = float(request.form['DTIRatio'])
    Education = float(request.form['Education'])
    EmploymentType = float(request.form['EmploymentType'])
    MaritalStatus = float(request.form['MaritalStatus'])
    HasMortgage = float(request.form['HasMortgage'])
    HasDependents = float(request.form['HasDependents'])
    LoanPurpose = float(request.form['LoanPurpose'])
    HasCoSigner = float(request.form['HasCoSigner'])

    # Feature engineering (same as training)
    EMI_Ratio = LoanAmount / Income

    input_data = pd.DataFrame([[
        Age, Income, LoanAmount, CreditScore, MonthsEmployed,
        NumCreditLines, InterestRate, LoanTerm, DTIRatio,
        Education, EmploymentType, MaritalStatus,
        HasMortgage, HasDependents, LoanPurpose,
        HasCoSigner, EMI_Ratio
    ]], columns=columns)

    prediction = model.predict(input_data)[0]

    result = "High Risk of Default" if prediction == 1 else "Low Risk of Default"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)