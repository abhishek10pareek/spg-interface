from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import os, dotenv
import xgboost

app = Flask(__name__)

with open('spg_pred/model/RandomForestClassifier.pickle', 'rb') as f:
    model = pickle.load(f)
with open('spg_pred/model/spacegroup_encoder.pickle', 'rb') as f:
    spacegroup_encoder = pickle.load(f)
with open('spg_pred/model/elements_encoder.pickle', 'rb') as f:
    element_encoder = pickle.load(f)


def featurize(formula, a, b, c, alpha, beta, gamma):
    List = [a, b, c, alpha, beta, gamma]

    for name, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if count == "":
            count = 1
        else:
            count = int(count)
        List.extend([1 + element_encoder.transform([name])[0], count])

    while (len(List) < 52):  # if using model trained on oqmd then replace 20 with 26
        List.append(0)

    input_data = np.array(List).reshape(1, -1)
    return input_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    formula = data['formula']
    a = float(data['a'])
    b = float(data['b'])
    c = float(data['c'])
    alpha = float(data['alpha'])
    beta = float(data['beta'])
    gamma = float(data['gamma'])

    features = featurize(formula, a, b, c, alpha, beta, gamma)
    probs = model.predict_proba(features)[0]

    # Get top-3 indices
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_spacegroups = spacegroup_encoder.inverse_transform(top3_indices)
    top3_probs = probs[top3_indices]

    # Build response
    results = [{"spacegroup": sg, "probability": round(float(p), 3)}
               for sg, p in zip(top3_spacegroups, top3_probs)]

    return jsonify({"top_3": results})


if __name__ == '__main__':
    app.run(debug=True)