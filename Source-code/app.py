from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model only once
model = pickle.load(open("placement.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    cgpa = float(data["cgpa"])

    prediction = model.predict(np.array([[cgpa]]))[0]

    return jsonify({
        "prediction": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)