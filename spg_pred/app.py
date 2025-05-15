from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import os
import requests

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


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def extract_file_id_from_url(url):
    import re
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract file ID from URL: {url}")


def download_if_needed(url, path):
    if not os.path.exists(path):
        if url is None:
            raise ValueError(f"URL for {path} is None, please set the environment variable properly.")
        file_id = extract_file_id_from_url(url)
        print(f"Downloading {path} from Google Drive file ID: {file_id}")
        download_file_from_google_drive(file_id, path)



download_if_needed(os.getenv("MODEL_URL"), MODEL_PATH)
download_if_needed(os.getenv("SPACEGROUP_ENCODER_URL"), SPACEGROUP_ENCODER_PATH)
download_if_needed(os.getenv("ELEMENTS_ENCODER_URL"), ELEMENTS_ENCODER_PATH)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SPACEGROUP_ENCODER_PATH, 'rb') as f:
    spacegroup_encoder = pickle.load(f)
with open(ELEMENTS_ENCODER_PATH, 'rb') as f:
    elements_encoder = pickle.load(f)


def featurize(formula, a, b, c, alpha, beta, gamma):
    List = [a, b, c, alpha, beta, gamma]

    for name, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if count == "":
            count = 1
        else:
            count = int(count)
        List.extend([1 + elements_encoder.transform([name])[0], count])

    while len(List) < 52:  # Adjust based on model training
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

    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_spacegroups = spacegroup_encoder.inverse_transform(top3_indices)
    top3_probs = probs[top3_indices]

    results = [{"spacegroup": sg, "probability": round(float(p), 3)}
               for sg, p in zip(top3_spacegroups, top3_probs)]

    return jsonify({"top_3": results})


if __name__ == '__main__':
    app.run(debug=True)
