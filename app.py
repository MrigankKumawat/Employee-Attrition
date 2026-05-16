from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Columns the fixed pipeline was trained on (after drops)
FEATURE_COLS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Age":                      [int(request.form["Age"])],
        "BusinessTravel":           [request.form["BusinessTravel"]],
        "DailyRate":                [int(request.form["DailyRate"])],
        "Department":               [request.form["Department"]],
        "DistanceFromHome":         [int(request.form["DistanceFromHome"])],
        "Education":                [int(request.form["Education"])],
        "EnvironmentSatisfaction":  [int(request.form["EnvironmentSatisfaction"])],
        "Gender":                   [request.form["Gender"]],
        "HourlyRate":               [int(request.form["HourlyRate"])],
        "JobInvolvement":           [int(request.form["JobInvolvement"])],
        "JobLevel":                 [int(request.form["JobLevel"])],
        "JobRole":                  [request.form["JobRole"]],
        "JobSatisfaction":          [int(request.form["JobSatisfaction"])],
        "MaritalStatus":            [request.form["MaritalStatus"]],
        "MonthlyIncome":            [int(request.form["MonthlyIncome"])],
        "MonthlyRate":              [int(request.form["MonthlyRate"])],
        "NumCompaniesWorked":       [int(request.form["NumCompaniesWorked"])],
        "OverTime":                 [request.form["OverTime"]],
        "PercentSalaryHike":        [int(request.form["PercentSalaryHike"])],
        "PerformanceRating":        [int(request.form["PerformanceRating"])],
        "RelationshipSatisfaction": [int(request.form["RelationshipSatisfaction"])],
        "TotalWorkingYears":        [int(request.form["TotalWorkingYears"])],
        "TrainingTimesLastYear":    [int(request.form["TrainingTimesLastYear"])],
        "WorkLifeBalance":          [int(request.form["WorkLifeBalance"])],
        "YearsSinceLastPromotion":  [int(request.form["YearsSinceLastPromotion"])],
        "YearsWithCurrManager":     [int(request.form["YearsWithCurrManager"])],
    }

    df = pd.DataFrame(data)[FEATURE_COLS]
    prepared = pipeline.transform(df)
    prediction = model.predict(prepared)[0]
    probability = model.predict_proba(prepared)[0][1]

    result = "leaving" if prediction == 1 else "staying"
    confidence = round(probability * 100, 1)

    return render_template("index.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)