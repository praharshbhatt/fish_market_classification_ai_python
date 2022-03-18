from joblib import dump, load
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
classifier = load("fish_classification_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    if prediction == 0:
        prediction = "Bream"
    elif prediction == 1:
        prediction = "Perch"
    elif prediction == 2:
        prediction = "Pike"
    elif prediction == 3:
        prediction = "Roach"
    elif prediction == 4:
        prediction = "Smelt"
    elif prediction == 5:
        prediction = "Whitefish"

    return render_template(
        "index.html",
        prediction_text="The fish belongs to species {}".format(prediction),
    )


if __name__ == "__main__":
    app.run()
