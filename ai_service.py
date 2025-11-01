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
EMB_CACHE = None  # Embedding cache
MODEL_READY = False  # ✅ Flag to track if ML assets are loaded


# --- ASSET LOADING (runs in background thread) ---
def load_assets():
    """Loads ML models, tokenizer, and embeddings in the background."""
    global TOKENIZER, MODEL, CLASSIFIER, EMB_CACHE, MODEL_READY

    print("--- Starting ML Asset Loading (This may take a moment) ---")
    try:
        base_dir = os.path.dirname(__file__)

        # 1. Load BioBERT model + tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        MODEL = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
        print(f"✅ Loaded BioBERT: {BIOBERT_MODEL_NAME}")

        # 2. Load Classifier
        CLASSIFIER = joblib.load(os.path.join(base_dir, MODEL_FILE))
        print(f"✅ Loaded classifier: {MODEL_FILE}")

        # 3. Load embeddings
        EMB_CACHE = joblib.load(os.path.join(base_dir, EMBEDDING_FILE))
        print(f"✅ Loaded embeddings: {EMBEDDING_FILE}")

        MODEL_READY = True
        print("--- ✅ All ML Assets Loaded Successfully ---")

    except Exception as e:
        print(f"❌ FATAL ERROR loading ML assets: {e}")
        MODEL_READY = False


# --- FEATURE PIPELINE ---
def get_embedding(text):
    global TOKENIZER, MODEL
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = MODEL(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def pair_embedding(drug1, drug2, emb_cache):
    drug1_key, drug2_key = drug1.upper(), drug2.upper()
    if drug1_key not in emb_cache:
        emb_cache[drug1_key] = get_embedding(drug1)
    if drug2_key not in emb_cache:
        emb_cache[drug2_key] = get_embedding(drug2)
    return np.concatenate([emb_cache[drug1_key], emb_cache[drug2_key]])


def predict_interaction(drug1, drug2):
    global CLASSIFIER, EMB_CACHE
    features = pair_embedding(drug1, drug2, EMB_CACHE).reshape(1, -1)
    return str(CLASSIFIER.predict(features)[0])


# --- FLASK APP ---
app = flask.Flask(__name__)

@app.route("/status", methods=["GET"])
def status():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        "status": "ready" if MODEL_READY else "loading"
    }), 200


@app.route('/check-ddi', methods=['POST'])
def check_ddi():
    global MODEL_READY, CLASSIFIER
    if not MODEL_READY:
        return jsonify({"message": "Model is still loading, please try again in a moment."}), 503

    try:
        data = request.json
        drugA = data.get("drugA", "")
        drugB = data.get("drugB", "")

        if not drugA or not drugB:
            return jsonify({"message": "Missing drugA or drugB"}), 400

        prediction_result = predict_interaction(drugA, drugB)
        print(f"--- RAW PREDICTION: {prediction_result} ---")

        severity = "high" if prediction_result.lower() != '0' else "low"
        details = {
            "severity": severity,
            "sideEffects": [
                f"Possible interaction: {prediction_result}" if severity == "high" else "No severe interaction detected."
            ],
            "recommendations": [
                "Stop use and consult doctor." if severity == "high" else "Continue as prescribed."
            ],
            "interactingMedicines": [drugA, drugB] if severity == "high" else []
        }

        return jsonify(details), 200

    except Exception as e:
        print(f"Prediction crashed: {e}")
        return jsonify({
            "message": "Internal error",
            "error": str(e)
        }), 500


# --- STARTUP ---
if __name__ == '__main__':
    print("🚀 Starting Flask server on port 8080...")
    # Start loading assets in a background thread (non-blocking)
    threading.Thread(target=load_assets, daemon=True).start()
    # Start Flask
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
