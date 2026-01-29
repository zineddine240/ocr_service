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
LOCATION = "us-central1" # 'global' n'est pas supporté pour ce modèle 

print(f"--- Initialisation Vertex AI ({PROJECT_ID}) ---")

model = None

def init_vertex():
    global model
    try:
        # 1. Tentative d'initialisation via Application Default Credentials (ADC)
        # Indispensable pour Google Cloud Run.
        try:
            from google import auth
            credentials, project = auth.default()
            print(f"trying to use ADC (Project: {project})...")
            vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
            model_name = "gemini-2.5-flash"
            model = GenerativeModel(model_name)
            print(f"✅ Modèle Vertex AI ({model_name}) chargé via ADC.")
            return True, ""
        except Exception as adc_err:
            print(f"ℹ️ ADC non disponible (Attendu en local si non connecté via gcloud) : {adc_err}")
            print("🔄 Passage à l'initialisation manuelle (Service Account Key)...")

        # 2. Reconstitution manuelle depuis les variables d'environnement (Fallback pour local/Render)
        private_key = os.getenv("GOOGLE_PRIVATE_KEY", "")
        if not private_key:
            return False, "La variable GOOGLE_PRIVATE_KEY est vide ou absente."
            
        import re
        # Nettoyage RADICAL de la clé (gestion des \n, espaces, etc.)
        private_key = private_key.replace('\\\\n', ' ').replace('\\n', ' ').replace('\\r', ' ')
        
        header = "-----BEGIN PRIVATE KEY-----"
        footer = "-----END PRIVATE KEY-----"
        
        content = private_key
        if header in private_key and footer in private_key:
            content = private_key.split(header)[1].split(footer)[0]
        
        content = re.sub(r'[^A-Za-z0-9+/=]', '', content)
        missing_padding = len(content) % 4
        if missing_padding:
            content += '=' * (4 - missing_padding)
        
        lines = [content[i:i+64] for i in range(0, len(content), 64)]
        private_key = f"{header}\n" + "\n".join(lines) + f"\n{footer}\n"

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
        
        model_name = "gemini-2.5-flash" 
        print(f"⏳ Chargement du modèle {model_name}...")
        model = GenerativeModel(model_name)
        print("✅ Modèle Vertex AI chargé avec succès via Service Account.")
        return True, ""
    except Exception as e:
        print("❌ ERREUR INITIALISATION VERTEX :")
        print(e)
        return False, str(e)

# Initialisation au démarrage
init_vertex()

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/scan', methods=['POST'])
def scan_image():
    global model
    if model is None:
        success, error_msg = init_vertex()
        if not success:
            return jsonify({"success": False, "error": f"Initialisation Vertex échouée: {error_msg}"}), 500

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
