from flask import Flask, request, jsonify
import threading
import pickle
import os

app = Flask(__name__)

ddi_model = None
drug_embeddings = None

def load_assets():
    """Loads ML assets (model and embeddings) in the background."""
    global ddi_model, drug_embeddings
    print("--- Starting ML Asset Loading (This may take a moment) ---")
    try:
        with open("ddi_model.pkl", "rb") as f:
            ddi_model = pickle.load(f)
        with open("drug_embeddings.pkl", "rb") as f:
            drug_embeddings = pickle.load(f)
        print("✅ Model and embeddings loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading assets: {e}")

# Start background loading so Flask can boot immediately
threading.Thread(target=load_assets, daemon=True).start()

@app.route('/')
def home():
    """Root route for quick health checks."""
    return jsonify({"message": "AI Model Service is running!"})

@app.route('/check-ddi', methods=['POST'])
def check_ddi():
    """Endpoint to check drug-drug interaction."""
    global ddi_model, drug_embeddings

    if ddi_model is None or drug_embeddings is None:
        return jsonify({"error": "Model is still loading, please try again in a few seconds."}), 503

    data = request.get_json()
    drugA = data.get("drugA")
    drugB = data.get("drugB")

    if not drugA or not drugB:
        return jsonify({"error": "Both 'drugA' and 'drugB' are required."}), 400

    # Dummy prediction (replace with your actual ML logic)
    result = f"Potential interaction between {drugA} and {drugB} detected."
    return jsonify({"result": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
