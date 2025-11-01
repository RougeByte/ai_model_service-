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

# Global assets
TOKENIZER = None
MODEL = None  # BioBERT Model
CLASSIFIER = None  # Your DDI Classifier (from ddi_model.pkl)
EMB_CACHE = None  # Your embedding cache (from drug_embeddings.pkl)

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
        print(f"Successfully loaded BioBERT model: {BIOBERT_MODEL_NAME}")
        
        # 2. Load the DDI Classifier
        CLASSIFIER = joblib.load(os.path.join(base_dir, MODEL_FILE))
        print(f"Successfully loaded classifier: {MODEL_FILE}")
        
        # 3. Load the Embedding Cache
        # CRITICAL FIX: The file name is now correct.
        EMB_CACHE = joblib.load(os.path.join(base_dir, EMBEDDING_FILE))
        print(f"Successfully loaded embeddings cache: {EMBEDDING_FILE}")
        
        print("--- All ML Assets Loaded Successfully ---")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Required file not found: {e}")
        # Raising an error prevents the Flask app from starting if assets are missing.
        raise RuntimeError("ML asset loading failed. Check file paths.")
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred during asset loading: {e}")
        raise RuntimeError("ML asset loading failed.")

# Initialize all assets upon startup
try:
    load_assets()
except RuntimeError:
    pass 

# --- FEATURE GENERATION PIPELINE ---

def get_embedding(text):
    """Generates BioBERT embedding for a single drug name."""
    # Access global variables
    global TOKENIZER, MODEL
    
    # Ensure text is clean and in the required format
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        # Get the output from the BioBERT model
        outputs = MODEL(**inputs)
    # Extract the CLS token embedding (first token output)
    return outputs.last_hidden_state[:,0,:].squeeze().numpy()

def pair_embedding(drug1, drug2, emb_cache):
    """Retrieves embeddings from cache or generates them, then concatenates the pair."""
    
    # CRITICAL FIX: Standardize drug names to UPPERCASE for cache lookup
    drug1_key = drug1.upper()
    drug2_key = drug2.upper()
    
    # Check cache for drug 1
    if drug1_key not in emb_cache:
        # If not found, generate new embedding
        emb_cache[drug1_key] = get_embedding(drug1)
    
    # Check cache for drug 2
    if drug2_key not in emb_cache:
        # If not found, generate new embedding
        emb_cache[drug2_key] = get_embedding(drug2)
        
    # Concatenate the two embedding vectors
    return np.concatenate([emb_cache[drug1_key], emb_cache[drug2_key]])

def predict_interaction(drug1, drug2):
    """Prepares features and runs prediction on the DDI classifier."""
    
    global CLASSIFIER, EMB_CACHE
    
    if CLASSIFIER is None:
        raise RuntimeError("Classifier not loaded.")
        
    # 1. Generate/retrieve combined features using the BioBERT pipeline
    # Pass the EMB_CACHE object which contains the embeddings
    features = pair_embedding(drug1, drug2, EMB_CACHE).reshape(1, -1)
    
    # 2. Run the classification model
    pred_result = CLASSIFIER.predict(features)[0]
    
    # FINAL FIX: Convert the NumPy/custom output to a standard Python string immediately
    return str(pred_result) 


# --- FLASK APPLICATION SETUP & API ENDPOINT ---
app = flask.Flask(__name__)

@app.route('/check-ddi', methods=['POST'])
def check_ddi():
    if CLASSIFIER is None:
        return jsonify({"message": "ML Model failed to load at startup."}), 500

    drugA = ""
    drugB = ""

    try:
        data = request.json
        # Normalize input drugs immediately
        drugA = data.get('drugA', '')
        drugB = data.get('drugB', '')
        
        if not drugA or not drugB:
            return jsonify({"message": "Missing drugA or drugB in request payload."}), 400

        # Run prediction
        prediction_result = predict_interaction(drugA, drugB)
        
        # ADDED DEBUGGING LOG: Prints the raw prediction result
        print(f"--- RAW PREDICTION RESULT (TYPE: {type(prediction_result).__name__}): {prediction_result} ---")
        
        # --- RESULT MAPPING ---
        
        severity = "low"
        interaction_type = ""
        
        # Check if the result is a non-zero string (meaning an interaction type was classified)
        pred_str = prediction_result.lower()

        if pred_str != '0':
            interaction_type = pred_str
            severity = "high"
        
        # --- CONSTRUCT FINAL RESPONSE DETAILS ---

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
        print(f"Prediction logic crashed during request processing: {e}")
        # Return a 500 status with the model error details
        return jsonify({
            "severity": "low",
            "sideEffects": [f"Internal Model Error: {str(e)}. Check Python logs."],
            "recommendations": ["Prediction failed. Check drug names and spelling."],
            "interactingMedicines": [drugA, drugB]
        }), 500


# --- SERVER STARTUP ---
if __name__ == '__main__':
    # CRITICAL FIX: Binding to 0.0.0.0 allows Node.js to communicate
    print("Starting Flask server...")
    app.run(debug=True, port=5000, host='0.0.0.0') 
