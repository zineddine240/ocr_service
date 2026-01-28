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
# On utilise le nom alias standard pour Gemini 3 Flash
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
        
        # Test du modèle spécifié
        try:
            # On vérifie si gemini-3-flash-preview est prêt, sinon on tente un alias alternatif ou on replie
            print(f"Tentative de chargement de {model_name}...")
            model = GenerativeModel(model_name)
            return True, "OK"
        except Exception as e:
            print(f"Erreur modèle {model_name}, tentative de repli...")
            model_name = "gemini-1.5-flash"
            model = GenerativeModel(model_name)
            return True, f"Modèle {model_name} utilisé par défaut."

    except Exception as e:
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
        "model": model_name, 
        "init_success": success, 
        "init_message": message
    })

@app.route('/scan', methods=['POST'])
def scan_image():
    global model, success, message
    
    # Tentative d'instanciation au moment de l'appel pour attraper l'erreur 404 de Google
    try:
        if not model:
            init_vertex()
        
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "Aucune image reçue"}), 400

        file = request.files['image']
        img_bytes = file.read()
        mime = file.content_type or "image/jpeg"
        if img_bytes.startswith(b'\x89PNG'): mime = 'image/png'
        
        image_part = Part.from_data(data=img_bytes, mime_type=mime)
        prompt = "1. Extract all text from this image, without any comments or explanations."

        # Appel au modèle
        print(f"📡 Appel Vertex AI avec {model_name}...")
        response = model.generate_content(
            [image_part, prompt],
            generation_config={"temperature": 0, "max_output_tokens": 8192}
        )

        return jsonify({"success": True, "text": response.text})

    except Exception as e:
        error_str = str(e)
        # Si c'est une erreur 404 (modèle non trouvé), on bascule immédiatement sur 1.5
        if "404" in error_str or "not found" in error_str.lower():
            print("⚠️ Modèle non trouvé par Google, bascule forcée sur 1.5-flash")
            try:
                fallback_model = GenerativeModel("gemini-1.5-flash")
                # Ré-essai immédiat du scan avec 1.5
                file.seek(0)
                img_bytes = file.read()
                image_part = Part.from_data(data=img_bytes, mime_type=mime)
                response = fallback_model.generate_content([image_part, prompt])
                return jsonify({"success": True, "text": response.text, "info": "Bascule auto sur 1.5 car 3 est indisponible"})
            except Exception as e2:
                return jsonify({"success": False, "error": f"Erreur critique suite à bascule: {str(e2)}"}), 500
        
        return jsonify({"success": False, "error": f"Erreur OCR: {error_str}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)