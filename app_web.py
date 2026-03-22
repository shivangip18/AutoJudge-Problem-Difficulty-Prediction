from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import numpy as np

from feature_utils import TextExtractor, HandcraftedTransformer, explain_prediction

app = Flask(__name__)

# Load trained models
clf = joblib.load("models/autojudge_classifier.joblib")
reg = joblib.load("models/autojudge_regressor.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form.get("title", "")
        description = request.form.get("description", "")
        input_desc = request.form.get("input_description", "")
        output_desc = request.form.get("output_description", "")

        df = pd.DataFrame(
            [
                {
                    "title": title,
                    "description": description,
                    "input_description": input_desc,
                    "output_description": output_desc,
                    "sample_io": "",
                }
            ]
        )

        pred_class = clf.predict(df)[0]
        pred_score = reg.predict(df)[0]
        confidence = round(np.max(clf.predict_proba(df)[0]) * 100, 2)

        explanations = explain_prediction(df)

        return redirect(
            url_for(
                "index",
                cls=pred_class,
                score=round(pred_score, 2),
                conf=confidence,
                exp=" | ".join(explanations),
            )
        )

    # GET request (after refresh or redirect)
    predicted_class = request.args.get("cls")
    predicted_score = request.args.get("score")
    confidence = request.args.get("conf")
    explanations = request.args.get("exp")

    return render_template(
        "index.html",
        predicted_class=predicted_class,
        predicted_score=predicted_score,
        confidence=confidence,
        explanations=explanations,
    )


if __name__ == "__main__":
    app.run(debug=True)
