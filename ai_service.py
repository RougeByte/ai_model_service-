import os
import pickle
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# CONFIGURATION
# ------------------------------
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MODEL_FILE = "ddi_model.pkl"
EMBEDDING_FILE = "drug_embeddings.pkl"

# ------------------------------
# INITIALIZE FLASK APP
# ------------------------------
app = Flask(__name__)

# ------------------------------
# LOAD MODEL AND ASSETS
# ------------------------------
print("🚀 Starting AI Model Service...")

try:
    # Load BioBERT
    print("🔹 Loading BioBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)

    # Load trained classifier and embeddings
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILE)
    emb_path = os.path.join(base_dir, EMBEDDING_FILE)

    print("🔹 Loading DDI classifier and embeddings...")
    with open(model_path, "rb") as f:
        ddi_model = pickle.load(f)
    with open(emb_path, "rb") as f:
        drug_embeddings = pickle.load(f)

    assets_loaded = True
    print("✅ All models loaded successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    assets_loaded = False


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def get_embedding(text):
    """Generate a BioBERT embedding for a given drug name."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def pair_embedding(drug1, drug2):
    """Generate concatenated embeddings for two drugs."""
    d1, d2 = drug1.upper(), drug2.upper()

    if d1 in drug_embeddings and d2 in drug_embeddings:
        emb1, emb2 = drug_embeddings[d1], drug_embeddings[d2]
    else:
        emb1, emb2 = get_embedding(drug1), get_embedding(drug2)

    return np.concatenate([emb1, emb2])


def predict_interaction(drug1, drug2):
    """Predicts the interaction type between two drugs."""
    features = pair_embedding(drug1, drug2).reshape(1, -1)
    pred = ddi_model.predict(features)[0]
    return str(pred)


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    return jsonify({"message": "AI Model Service is running!"})


@app.route("/health")
def health():
    if assets_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "error"}), 500


@app.route("/check-ddi", methods=["POST"])
def check_ddi():
    if not assets_loaded:
        return jsonify({"error": "Model not loaded yet"}), 503

    try:
        data = request.get_json(force=True)
        drugA = data.get("drugA")
        drugB = data.get("drugB")

        if not drugA or not drugB:
            return jsonify({"error": "Both 'drugA' and 'drugB' are required"}), 400

        result = predict_interaction(drugA, drugB)
        print(f"🔍 Predicted interaction: {result}")

        if result != "0":
            return jsonify({
                "severity": "high",
                "interactionType": result,
                "sideEffects": [
                    "Severe interaction detected.",
                    "Requires immediate medical attention."
                ],
                "recommendations": [
                    "Stop usage immediately.",
                    "Consult a doctor before continuing."
                ],
                "interactingMedicines": [drugA, drugB]
            })
        else:
            return jsonify({
                "severity": "low",
                "sideEffects": [
                    "No severe interaction detected.",
                    "Monitor for mild symptoms."
                ],
                "recommendations": [
                    "Continue as prescribed."
                ]
            })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
