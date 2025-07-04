import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ── load the (pipeline) model once at start-up ──────────────────────
model = joblib.load("artifacts/co2_model_light.pkl")      # adjust path if needed

# ── routes ──────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template("index.html")        # landing page

@app.route('/Prediction', methods=['GET'])
def prediction_form():                          # renamed for clarity
    return render_template("index1.html")       # form page

@app.route('/predict', methods=['POST'])
def predict():
    # 1️⃣  grab form fields (all come in as strings)
    country_name   = request.form['country_name']
    country_code   = request.form['country_code']
    indicator_name = request.form['indicator_name']
    indicator_code = request.form['indicator_code']
    year           = float(request.form['year'])   # cast to numeric

    # 2️⃣  build the one-row DataFrame exactly like training
    input_df = pd.DataFrame([[country_name,
                              country_code,
                              indicator_name,
                              indicator_code,
                              year]],
                            columns=['CountryName',
                                     'CountryCode',
                                     'IndicatorName',
                                     'IndicatorCode',
                                     'Year'])

    # 3️⃣  predict (the model may include its own encoder)
    result = model.predict(input_df)[0]

    # 4️⃣  show the result
    return render_template("result.html", prediction=result)

# ── launch ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
