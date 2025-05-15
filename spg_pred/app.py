from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import os
import dotenv
import requests


dotenv.load_dotenv()

app = Flask(__name__)


MODEL_PATH = "model/RandomForestClassifier.pickle"
SPACEGROUP_ENCODER_PATH = "model/spacegroup_encoder.pickle"
ELEMENTS_ENCODER_PATH = "model/elements_encoder.pickle"


os.makedirs("model", exist_ok=True)

def download_if_needed(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download from {url}. Status code: {response.status_code}")


download_if_needed(os.getenv("MODEL_URL"), MODEL_PATH)
download_if_needed(os.getenv("SPACEGROUP_ENCODER_URL"), SPACEGROUP_ENCODER_PATH)
download_if_needed(os.getenv("ELEMENT_ENCODER_URL"), ELEMENTS_ENCODER_PATH)


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SPACEGROUP_ENCODER_PATH, 'rb') as f:
    spacegroup_encoder = pickle.load(f)
with open(ELEMENTS_ENCODER_PATH, 'rb') as f:
    element_encoder = pickle.load(f)


def featurize(formula, a, b, c, alpha, beta, gamma):
    features = [a, b, c, alpha, beta, gamma]
    for name, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        count = int(count) if count else 1
        encoded = element_encoder.transform([name])[0]
        features.extend([1 + encoded, count])
    while len(features) < 52:
        features.append(0)
    return np.array(features).reshape(1, -1)

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

    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_spacegroups = spacegroup_encoder.inverse_transform(top3_indices)
    top3_probs = probs[top3_indices]

    results = [{"spacegroup": sg, "probability": round(float(p), 3)}
               for sg, p in zip(top3_spacegroups, top3_probs)]

    return jsonify({"top_3": results})

if __name__ == '__main__':
    app.run(debug=True)
