from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

with open("models/logistic_model.pkl", "rb") as file:
    logistic_model = pickle.load(file)

with open("models/svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # GET USER INPUT
        # -------------------------------------------------

        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        age = float(request.form["age"])


        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [[
                glucose,
                blood_pressure,
                insulin,
                bmi,
                age
            ]],
            columns=[
                "Glucose",
                "Blood Pressure",
                "Insulin",
                "BMI",
                "Age"
            ]
        )


        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        input_scaled = scaler.transform(input_data)


        # -------------------------------------------------
        # LOGISTIC REGRESSION PROBABILITY
        # -------------------------------------------------

        logistic_probability = (
            logistic_model
            .predict_proba(input_scaled)[0][1]
        )


        # -------------------------------------------------
        # SVM PROBABILITY
        # -------------------------------------------------

        svm_probability = (
            svm_model
            .predict_proba(input_scaled)[0][1]
        )


        # -------------------------------------------------
        # COMBINE BOTH MODELS
        # -------------------------------------------------

        combined_probability = (
            logistic_probability +
            svm_probability
        ) / 2


        # Convert to percentage

        combined_probability_percent = round(
            combined_probability * 100,
            2
        )


        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        if combined_probability >= 0.50:

            final_result = "Diabetes Detected"

        else:

            final_result = "No Diabetes"


        # -------------------------------------------------
        # SEND RESULT TO FRONTEND
        # -------------------------------------------------

        return render_template(
            "index.html",

            final_result=final_result,

            combined_probability=
                combined_probability_percent,

            glucose=glucose,

            blood_pressure=blood_pressure,

            insulin=insulin,

            bmi=bmi,

            age=age
        )


    except Exception as error:

        return render_template(
            "index.html",
            error=str(error)
        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )