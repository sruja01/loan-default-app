from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
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

    # Feature engineering (IMPORTANT — same as training)
    EMI_Ratio = LoanAmount / Income

    # Create dataframe
    input_data = pd.DataFrame([[
        Age, Income, LoanAmount, CreditScore, MonthsEmployed,
        NumCreditLines, InterestRate, LoanTerm, DTIRatio,
        Education, EmploymentType, MaritalStatus,
        HasMortgage, HasDependents, LoanPurpose,
        HasCoSigner, EMI_Ratio
    ]], columns=columns)

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "High Risk of Default"
    else:
        result = "Low Risk of Default"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
