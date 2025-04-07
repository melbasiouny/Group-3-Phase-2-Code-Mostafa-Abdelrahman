import os
import joblib
import pandas as pd
import traceback # For detailed error logging
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
MODEL_DIR = os.path.dirname(os.path.realpath(__file__)) # Gets directory where app.py is
MODEL_PATH = os.path.join(MODEL_DIR, 'wildfire_prediction_pipeline.joblib')
model = None

print(f"Attempting to load model from: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    if hasattr(model, 'steps'):
         print(f"Model steps: {model.steps}")
except Exception as e:
    print(f"FATAL: Failed to load model on startup: {e}")
    print(traceback.format_exc())
    # Optional: Depending on requirements, you might want the app to fail startup
    # raise e
# --- End Model Loading ---


@app.route('/ping', methods=['GET'])
def ping():
    """ Basic health check endpoint """
    # Check if model loaded successfully during startup
    health = "healthy" if model is not None else "unhealthy: model not loaded"
    status = 200 if model is not None else 500
    return jsonify({"status": health}), status


@app.route('/predict', methods=['POST'])
def predict():
    """ Main prediction endpoint """
    if model is None:
        print("Error: Model is not loaded.")
        return jsonify({"error": "Model not loaded properly, check server logs."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        input_data = request.get_json()

        # Basic validation (expecting list of dicts)
        if not isinstance(input_data, list) or not all(isinstance(item, dict) for item in input_data):
            raise ValueError("Input must be a JSON list of feature dictionaries.")

        # Convert to DataFrame (ensure columns match training)
        input_df = pd.DataFrame(input_data)
        print(f"Received input for prediction (head):\n{input_df.head()}")

        # Make predictions
        predictions = model.predict(input_df)

        # Format response
        response = {"predictions": predictions.tolist()} # Convert numpy array
        return jsonify(response), 200

    except ValueError as ve:
        print(f"Input validation error: {ve}")
        return jsonify({"error": f"Invalid input format: {ve}"}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc()) # Log detailed traceback
        return jsonify({"error": "An error occurred during prediction."}), 500


# Run the app (for local testing - Gunicorn will run it in App Runner)
if __name__ == '__main__':
# Make sure to set the PORT environment variable for App Runner compatibility
     port = int(os.environ.get('PORT', 8080))
     app.run(host='0.0.0.0', port=port, debug=True) # debug=True for local testing ONLY