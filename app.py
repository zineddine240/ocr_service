import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

app = Flask(__name__)
# Autoriser les requêtes venant de n'importe où (pour le dev)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURATION VERTEX AI ---
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "us-central1"

print(f"--- Initialisation Vertex AI ({PROJECT_ID}) ---")

model = None

def init_vertex():
    global model
    try:
        # Reconstitution des identifiants depuis les variables d'environnement
        private_key = os.getenv("GOOGLE_PRIVATE_KEY", "")
        if private_key:
            private_key = private_key.replace('\\n', '\n').strip()
            if private_key.startswith('"') and private_key.endswith('"'):
                private_key = private_key[1:-1]

        credentials_info = {
            "type": "service_account",
            "project_id": PROJECT_ID,
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": private_key,
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
        
        # MODIFICATION : Utilisation de Gemini 2.5 Flash
        model_name = "gemini-2.5-flash" 
        
        print(f"⏳ Chargement du modèle {model_name}...")
        model = GenerativeModel(model_name)
        print("✅ Modèle Vertex AI chargé avec succès.")
        return True
    except Exception as e:
        print("❌ ERREUR INITIALISATION VERTEX :")
        print(e)
        return False

# Initialisation au démarrage
init_vertex()

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/scan', methods=['POST'])
def scan_image():
    global model
    if model is None:
        if not init_vertex():
            return jsonify({"success": False, "error": "Le modèle AI n'est pas chargé (Erreur serveur)"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400

    file = request.files['image']
    
    try:
        # Lecture de l'image
        img_bytes = file.read()
        
        # Préparation pour Vertex AI
        image_part = Part.from_data(
            data=img_bytes, 
            mime_type=file.content_type if file.content_type else "image/jpeg"
        )

        prompt = "Extract all text from this image exactly as it appears. No markdown, no comments."

        print("🚀 Envoi à Vertex AI (Gemini 2.5 Flash)...")
        start_time = time.time()
        
        response = None
        for attempt in range(4): # Try 4 times (Initial + 3 retries)
            try:
                response = model.generate_content([prompt, image_part])
                break
            except ResourceExhausted:
                if attempt == 3:
                    raise # Re-raise if last attempt
                wait_time = 2 ** attempt
                print(f"⚠️ Quota dépassé (429), nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                raise e

        final_text = response.text
        duration = time.time() - start_time
        print(f"✅ Réponse reçue en {duration:.2f}s !")
        
        return jsonify({
            "success": True, 
            "text": final_text.strip(),
            "time": f"{duration:.2f}s"
        })

    except Exception as e:
        print(f"❌ ERREUR PENDANT LE SCAN : {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
