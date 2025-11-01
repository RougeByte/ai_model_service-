import flask
import joblib
import os
import json
import numpy as np
import torch
import threading
from transformers import AutoTokenizer, AutoModel
from flask import request, jsonify

# --- CONFIGURATION & FILE PATHS ---

MODEL_FILE = 'ddi_model.pkl'
EMBEDDING_FILE = 'drug_embeddings.pkl'
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Global assets
TOKENIZER = None
MODEL = None  # BioBERT Model
CLASSIFIER = None  # DDI Classifier
EMB_CACHE = None  # Embedding Cache


# --- ASSET LOADING ---

def load_assets():
    """Loads all necessary ML models, tokenizer, and embedding cache into memory."""
    global TOKENIZER, MODEL, CLASSIFIER, EMB_CACHE

    base_dir = os.path.dirname(__file__)

    print("--- Starting ML Asset Loading (This may take a moment) ---")

    try:
        # 1. Load BioBERT Tokenizer and Model
        TOKENIZER = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        MODEL = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
        print(f"✅ Successfully loaded BioBERT model: {BIOBERT_MODEL_NAME}")

        # 2. Load the DDI Classifier
        CLASSIFIER = joblib.load(os.path.join(base_dir, MODEL_FILE))
        print(f"✅ Successfully loaded classifier: {MODEL_FILE}")

        # 3. Load the Embedding Cache
        EMB_CACHE = joblib.load(os.path.join(base_dir, EMBEDDING_FILE))
        print(f"✅ Successfully loaded embeddings cache: {EMBEDDING_FILE}")

        print("--- ✅ All ML Assets Loaded Successfully ---")

    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Required file not found: {e}")
        raise RuntimeError("ML asset loading failed. Check file paths.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Unexpected error during asset loading: {e}")
        raise RuntimeError("ML asset loading failed.")


# --- FEATURE GENERATION PIPELINE ---

def get_embedding(text):
    """Generates BioBERT embedding for a single drug name."""
    global TOKENIZER, MODEL
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = MODEL(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def pair_embedding(drug1, drug2, emb_cache):
    """Retrieves embeddings from cache or generates them, then concatenates the pair."""
    drug1_key = drug1.upper()
    drug2_key = drug2.upper()

    if drug1_key not in emb_cache:
        emb_cache[drug1_key] = get_embedding(drug1)
    if drug2_key not in emb_cache:
        emb_cache[drug2_key] = get_embedding(drug2)

    return np.concatenate([emb_cache[drug1_key], emb_cache[drug2_key]])


def predict_interaction(drug1, drug2):
    """Prepares features and runs prediction on the DDI classifier."""
    global CLASSIFIER, EMB_CACHE

    if CLASSIFIER is None:
        raise RuntimeError("Classifier not loaded yet.")

    features = pair_embedding(drug1, drug2, EMB_CACHE).reshape(1, -1)
    pred_result = CLASSIFIER.predict(features)[0]
    return str(pred_result)


# --- FLASK APPLICATION SETUP & ROUTES ---

app = flask.Flask(__name__)


@app.route('/status', methods=['GET'])
def status():
    """Check if ML models are ready."""
    global CLASSIFIER
    if CLASSIFIER is None:
        return jsonify({"status": "loading"}), 202
    else:
        return jsonify({"status": "ready"}), 200


@app.route('/check-ddi', methods=['POST'])
def check_ddi():
    global CLASSIFIER

    if CLASSIFIER is None:
        return jsonify({"message": "ML Model still loading. Try again in a few seconds."}), 503

    drugA = ""
    drugB = ""

    try:
        data = request.json
        drugA = data.get('drugA', '')
        drugB = data.get('drugB', '')

        if not drugA or not drugB:
            return jsonify({"message": "Missing drugA or drugB in request payload."}), 400

        prediction_result = predict_interaction(drugA, drugB)
        print(f"--- RAW PREDICTION RESULT: {prediction_result} ---")

        severity = "low"
        interaction_type = ""
        pred_str = prediction_result.lower()

        if pred_str != '0':
            interaction_type = pred_str
            severity = "high"

        if severity == "high":
            details = {
                "severity": "high",
                "sideEffects": [
                    f"Possible Severe Interaction ({interaction_type or 'Classification'}) detected.",
                    "Requires immediate medical review."
                ],
                "recommendations": ["STOP use immediately.", "Do not take concurrently."],
                "interactingMedicines": [drugA, drugB]
            }
        else:
            details = {
                "severity": "low",
                "sideEffects": ["No severe interaction detected by the model.", "Monitor for unusual symptoms."],
                "recommendations": ["Continue as prescribed.", "Take as directed."],
                "interactingMedicines": []
            }

        return jsonify(details), 200

    except Exception as e:
        print(f"❌ Prediction logic crashed: {e}")
        return jsonify({
            "severity": "low",
            "sideEffects": [f"Internal Model Error: {str(e)}. Check Python logs."],
            "recommendations": ["Prediction failed. Check drug names and spelling."],
            "interactingMedicines": [drugA, drugB]
        }), 500


# --- SERVER STARTUP (CLOUD RUN COMPATIBLE) ---

if __name__ == '__main__':
    print("🚀 Starting Flask server (models will load in background)...")

    # Background model loading (non-blocking)
    threading.Thread(target=load_assets, daemon=True).start()

    # Run Flask app on all interfaces (Cloud Run requires this)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
