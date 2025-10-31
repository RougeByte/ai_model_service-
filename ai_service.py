import flask
import joblib
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from flask import request, jsonify

# --- CONFIGURATION & FILE PATHS ---
MODEL_FILE = 'ddi_model.pkl'
EMBEDDING_FILE = 'drug_embeddings.pkl' 
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Global assets (lazy-loaded)
TOKENIZER = None
MODEL = None
CLASSIFIER = None
EMB_CACHE = None

# --- ASSET LOADING ---

def load_classifier_and_cache():
    """Load lightweight assets immediately (safe for 512MB)."""
    global CLASSIFIER, EMB_CACHE
    base_dir = os.path.dirname(__file__)

    print("--- Loading DDI Classifier and Embeddings (light) ---")
    CLASSIFIER = joblib.load(os.path.join(base_dir, MODEL_FILE))
    EMB_CACHE = joblib.load(os.path.join(base_dir, EMBEDDING_FILE))
    print("--- Light assets loaded successfully ---")

def get_biobert():
    """Lazy load BioBERT only when needed (first request)."""
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        print("⚙️ Loading BioBERT model (first request, may take a few seconds)...")
        TOKENIZER = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        MODEL = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
        MODEL.eval()
        print("✅ BioBERT loaded successfully.")
    return TOKENIZER, MODEL

# Load only the small assets on startup
load_classifier_and_cache()

# --- FEATURE GENERATION PIPELINE ---

def get_embedding(text):
    """Generates BioBERT embedding for a single drug name."""
    TOKENIZER, MODEL = get_biobert()
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = MODEL(**inputs)
    return outputs.last_hidden_state[:,0,:].squeeze().numpy()

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
    features = pair_embedding(drug1, drug2, EMB_CACHE).reshape(1, -1)
    pred_result = CLASSIFIER.predict(features)[0]
    return str(pred_result)

# --- FLASK APPLICATION SETUP & API ENDPOINT ---
app = flask.Flask(__name__)

@app.route('/check-ddi', methods=['POST'])
def check_ddi():
    global CLASSIFIER

    if CLASSIFIER is None:
        return jsonify({"message": "ML Model failed to load at startup."}), 500

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
        print(f"Prediction error: {e}")
        return jsonify({
            "severity": "low",
            "sideEffects": [f"Internal Error: {str(e)}"],
            "recommendations": ["Prediction failed. Check drug names and spelling."],
            "interactingMedicines": [drugA, drugB]
        }), 500

# --- SERVER STARTUP ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False, port=5000, host='0.0.0.0')
