"""
Flask API for Iris Sepal Length Prediction
===========================================
This API loads the trained RandomForest model and provides
prediction endpoints for sepal_length based on flower features.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Model path (from Docker volume)
MODEL_PATH = os.getenv('MODEL_PATH', '/models/model.pkl')

# Global model variable
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    try:
        print(f"📂 Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return True
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    return jsonify({
        'status': 'online',
        'service': 'Iris Prediction API',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': 'POST /predict',
            'health': 'GET /health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check for Docker/Kubernetes."""
    if model is not None:
        return jsonify({'status': 'healthy', 'model': 'loaded'}), 200
    else:
        return jsonify({'status': 'unhealthy', 'model': 'not loaded'}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sepal_length based on flower features.

    Expected JSON body:
    {
        "sepal_width": float,
        "petal_length": float (optional),
        "petal_width": float (optional)
    }

    Note: If only sepal_width is provided, other features default to 0 (normalized).
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Get features with defaults (0 = mean value after normalization)
        sepal_width = data.get('sepal_width')
        petal_length = data.get('petal_length', 0)
        petal_width = data.get('petal_width', 0)

        if sepal_width is None:
            return jsonify({'error': 'sepal_width is required'}), 400

        # Prepare features array
        features = np.array([[sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({
            'predicted_sepal_length': round(float(prediction), 2),
            'input': {
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 IRIS PREDICTION API (Flask)")
    print("=" * 60)

    # Load model at startup
    if load_model():
        # Run Flask app
        port = int(os.getenv('API_PORT', 5000))
        print(f"\n🌐 Starting server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to start API: Model not loaded")
        exit(1)
