import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import traceback

app = Flask(__name__)

# CORS avec support explicite pour les erreurs
CORS(app, resources={r"/*": {"origins": "*"}})

#--- CONFIGURATION VERTEX AI ---
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
# La documentation indique "global" pour Gemini 3 Flash Preview
LOCATION_GLOBAL = "global"
LOCATION_DEFAULT = "us-central1"

model_3 = None
model_15 = None

def init_vertex():
    global model_3, model_15
    try:
        pk = os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n').strip()
        if pk.startswith('"') and pk.endswith('"'): pk = pk[1:-1]

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
        
        # Initialisation pour Gemini 3 (Location global d'après la doc)
        print(f"Initialisation Vertex AI pour Gemini 3 (location: {LOCATION_GLOBAL})...")
        vertexai.init(project=PROJECT_ID, location=LOCATION_GLOBAL, credentials=creds)
        model_3 = GenerativeModel("gemini-3-flash-preview")
        
        # Initialisation secondaire pour Gemini 1.5 (Location us-central1 pour stabilité)
        # Note: vertexai.init est global au process, mais on peut changer la location si besoin
        # Mais essayons de voir si 1.5 fonctionne aussi en global ou si on gère la bascule au moment du scan.
        
        return True, "Initialisé avec succès"

    except Exception as e:
        print(f"Erreur d'initialisation: {str(e)}")
        return False, str(e)

# Initialisation au démarrage
success, message = init_vertex()

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "online", 
        "init_success": success, 
        "init_message": message
    })

@app.route('/scan', methods=['POST'])
def scan_image():
    global model_3, success
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400

    file = request.files['image']
    img_bytes = file.read()
    mime = file.content_type or "image/jpeg"
    if img_bytes.startswith(b'\x89PNG'): mime = 'image/png'
    
    prompt = "1. Extract all text from this image, without any comments or explanations."
    image_part = Part.from_data(data=img_bytes, mime_type=mime)

    # 1. Tentative avec Gemini 3 (Location Global)
    try:
        if model_3:
            print("� Tentative avec Gemini 3 Flash Preview (Global)...")
            response = model_3.generate_content(
                [image_part, prompt],
                generation_config={"temperature": 0, "max_output_tokens": 8192}
            )
            return jsonify({"success": True, "text": response.text, "model": "gemini-3-flash-preview"})
    except Exception as e:
        print(f"⚠️ Échec Gemini 3: {str(e)}")
        
    # 2. Bascule sur Gemini 1.5 Flash (Location us-central1)
    try:
        print("🔄 Bascule sur Gemini 1.5 Flash (Location standard)...")
        # On ré-init sur us-central1 pour le fallback
        pk = os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n').strip()
        if pk.startswith('"') and pk.endswith('"'): pk = pk[1:-1]
        
        creds = service_account.Credentials.from_service_account_info({
            "type": "service_account",
            "project_id": PROJECT_ID,
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": pk,
            "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        })
        
        vertexai.init(project=PROJECT_ID, location=LOCATION_DEFAULT, credentials=creds)
        model_15 = GenerativeModel("gemini-1.5-flash")
        
        # On doit re-créer les objets Part car l'init a changé la région
        image_part = Part.from_data(data=img_bytes, mime_type=mime)
        response = model_15.generate_content([image_part, prompt])
        
        return jsonify({"success": True, "text": response.text, "model": "gemini-1.5-flash", "info": "Gemini 3 indisponible"})
    
    except Exception as e:
        return jsonify({"success": False, "error": f"Erreur critique: {str(e)}", "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)