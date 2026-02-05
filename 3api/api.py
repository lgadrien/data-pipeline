"""
Flask API for Iris Sepal Length Prediction
===========================================
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = '/models/model.pkl'
METRICS_PATH = '/models/metrics.json'
model = None
metrics = None


def load_model():
    global model, metrics
    try:
        print(f"📂 Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded!")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            print("✅ Metrics loaded!")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'service': 'Iris Prediction API',
        'endpoints': ['/predict', '/metrics', '/health']
    })


@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'healthy'}), 200
    return jsonify({'status': 'unhealthy'}), 503


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Retourne les performances du modèle."""
    if metrics is None:
        return jsonify({'error': 'Metrics not available'}), 404
    return jsonify(metrics)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json()

        if not data or 'sepal_width' not in data:
            return jsonify({'error': 'sepal_width is required'}), 400

        sepal_width = float(data['sepal_width'])
        features = np.array([[sepal_width]])
        prediction = model.predict(features)[0]

        return jsonify({
            'predicted_sepal_length': round(float(prediction), 2),
            'input': {'sepal_width': sepal_width}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 IRIS PREDICTION API")
    if load_model():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        exit(1)
