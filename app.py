from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load your trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # IMPORTANT: names must match HTML exactly
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bloodpressure = float(request.form["bloodpressure"])
        skinthickness = float(request.form["skinthickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = float(request.form["age"])

        # model input
        features = np.array([[pregnancies, glucose, bloodpressure,
                              skinthickness, insulin, bmi,
                              dpf, age]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "⚠️ High Risk of Diabetes"
        else:
            result = "✅ Low Risk of Diabetes"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)