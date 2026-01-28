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
LOCATION = "us-central1"

model = None
model_name = "gemini-3-flash-preview" 

def init_vertex():
    global model, model_name
    try:
        # Vérification des variables d'environnement
        required_vars = ["GOOGLE_PROJECT_ID", "GOOGLE_PRIVATE_KEY", "GOOGLE_CLIENT_EMAIL", "GOOGLE_PRIVATE_KEY_ID", "GOOGLE_CLIENT_ID"]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            return False, f"Variables manquantes : {', '.join(missing)}"

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
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        
        # Test du modèle 3
        try:
            model = GenerativeModel(model_name)
            # On ne fait pas d'appel ici pour éviter les délais
            return True, "OK"
        except Exception as e:
            # Repli auto si Gemini 3 est indisponible
            model_name = "gemini-1.5-flash"
            model = GenerativeModel(model_name)
            return True, f"Modèle 3 indisponible, repli sur 1.5. Erreur: {str(e)}"

    except Exception as e:
        return False, str(e)

# Initialisation au démarrage
success, message = init_vertex()
print(f"--- Init Status: {success}, {message} ---")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "online", 
        "model": model_name, 
        "init_success": success, 
        "init_message": message
    })

@app.route('/scan', methods=['POST'])
def scan_image():
    global model, success, message
    
    if not success:
        success, message = init_vertex() # Ré-essai
        if not success:
            return jsonify({"success": False, "error": f"Initialisation échouée: {message}"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400

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

        if not response.candidates:
             return jsonify({"success": False, "error": "Google AI n'a retourné aucun résultat (bloqué par filtres ?)"}), 500

        # Accès sécurisé au texte
        try:
            extracted_text = response.text
        except Exception:
            extracted_text = response.candidates[0].content.parts[0].text if response.candidates else "Erreur lecture texte"

        return jsonify({"success": True, "text": extracted_text})

    except Exception as e:
        return jsonify({
            "success": False, 
            "error": f"Erreur OCR: {str(e)}", 
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)