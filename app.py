import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import traceback

app = Flask(__name__)

# CONFIGURATION CORS ULTRA-SIMPLE POUR LA PROD
CORS(app, resources={r"/*": {"origins": "*"}})

#--- CONFIGURATION VERTEX AI ---
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "us-central1"

print(f"--- Initialisation Vertex AI ({PROJECT_ID}) ---")

model = None
# On utilise gemini-2.5-flash comme demandé
model_name = "gemini-2.5-flash" 

def init_vertex():
    global model, model_name
    try:
        pk = os.getenv("GOOGLE_PRIVATE_KEY")
        if not pk:
            print("❌ ERREUR: GOOGLE_PRIVATE_KEY manquante")
            return
            
        pk = pk.replace('\\n', '\n').strip()
        if pk.startswith('"') and pk.endswith('"'):
            pk = pk[1:-1]

        credentials_info = {
            "type": "service_account",
            "project_id": PROJECT_ID,
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": pk,
            "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GOOGLE_CLIENT_EMAIL', '').replace('@', '%40')}",
            "universe_domain": "googleapis.com"
        }
        
        creds = service_account.Credentials.from_service_account_info(credentials_info)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        
        # Initialisation du modèle demandé
        try:
            print(f"⏳ Chargement du modèle {model_name}...")
            model = GenerativeModel(model_name)
            print(f"✅ Modèle {model_name} initialisé.")
        except Exception as e:
            print(f"⚠️ Erreur chargement {model_name}: {str(e)}")
            # Fallback sur 1.5 flash si 2.5 n'est pas dispo
            model_name = "gemini-1.5-flash"
            model = GenerativeModel(model_name)
            print(f"✅ Repli sur {model_name} réussi.")

    except Exception as e:
        print(f"❌ ERREUR INIT VERTEX: {str(e)}")

init_vertex()

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ready", "model": model_name})

@app.route('/scan', methods=['POST'])
def scan_image():
    if model is None:
        init_vertex()
        if model is None:
            return jsonify({"success": False, "error": "Vertex AI not initialized"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image"}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()
        
        mime = file.content_type or "image/jpeg"
        if img_bytes.startswith(b'\x89PNG'): mime = 'image/png'
        
        image_part = Part.from_data(data=img_bytes, mime_type=mime)
        prompt = "1. Extract all text from this image, without any comments or explanations."

        response = model.generate_content(
            [image_part, prompt],
            generation_config={"temperature": 0, "max_output_tokens": 8192}
        )

        return jsonify({"success": True, "text": response.text})

    except Exception as e:
        print(f"❌ ERROR SCAN: {str(e)}")
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)