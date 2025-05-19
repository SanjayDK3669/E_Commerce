from flask import Flask, render_template, request
import joblib
import numpy as np



# Load your trained model
model = joblib.load("model_joblib.pkl")

app = Flask(__name__)



def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])

def predict():
    prediction_text = None

    if request.method == "POST":
        try:
            avg_sess_len = float(request.form["Avg._Session_Length"])
            time_on_app = float(request.form["Time_on_App"])
            time_on_web = float(request.form["Time_on_Website"])
            length_of_membership = float(request.form["Length_of_Membership"])
        except ValueError:
            prediction_text = "Please enter valid numeric inputs."
            return render_template("index.html", prediction_text=prediction_text)

        features = np.array([[avg_sess_len, time_on_app, time_on_web, length_of_membership]])

        pred = model.predict(features)
        
        if pred[0] > 0:
            prediction_value = pred[0]
        else:
            prediction_value = 0.0

        prediction_text = f"{prediction_value:,.2f}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
