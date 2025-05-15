from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # load env vars from .env

app = Flask(__name__)

MODEL_PATH = "model/RandomForestClassifier.pickle"
SPACEGROUP_ENCODER_PATH = "model/spacegroup_encoder.pickle"
ELEMENTS_ENCODER_PATH = "model/elements_encoder.pickle"

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def download_if_needed(file_id, path):
    if not os.path.exists(path):
        print(f"Downloading {path} from Google Drive file ID: {file_id}")
        download_file_from_google_drive(file_id, path)

# Download model and encoders if missing
download_if_needed(os.getenv("MODEL_FILE_ID"), MODEL_PATH)
download_if_needed(os.getenv("SPACEGROUP_ENCODER_FILE_ID"), SPACEGROUP_ENCODER_PATH)
download_if_needed(os.getenv("ELEMENT_ENCODER_FILE_ID"), ELEMENTS_ENCODER_PATH)

# Load model and encoders
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SPACEGROUP_ENCODER_PATH, 'rb') as f:
    spacegroup_encoder = pickle.load(f)
with open(ELEMENTS_ENCODER_PATH, 'rb') as f:
    element_encoder = pickle.load(f)

def featurize(formula, a, b, c, alpha, beta, gamma):
    List = [a, b, c, alpha, beta, gamma]

    for name, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if count == "":
            count = 1
        else:
            count = int(count)
        List.extend([1 + element_encoder.transform([name])[0], count])

    while len(List) < 52:  # adjust if needed based on your training
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

    results = [{"spacegroup": sg, "probability": round(float(p), 3)}
               for sg, p in zip(top3_spacegroups, top3_probs)]

    return jsonify({"top_3": results})

if __name__ == '__main__':
    app.run(debug=True)
