import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import traceback

app = Flask(__name__)

# CONFIGURATION CORS ROBUSTE pour Render
# On autorise tout le monde (*) et on désactive supports_credentials pour éviter les erreurs CORS sur 500
CORS(app, resources={r"/*": {"origins": "*"}})

#--- CONFIGURATION VERTEX AI ---
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "us-central1"

print(f"--- Initialisation Vertex AI ({PROJECT_ID}) ---")

model = None
model_name = "gemini-2.5-pro"

def init_vertex():
    global model, model_name
    try:
        private_key = os.getenv("GOOGLE_PRIVATE_KEY")
        if private_key:
            private_key = private_key.replace('\\n', '\n').strip()
            if private_key.startswith('"') or private_key.startswith("'"):
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
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GOOGLE_CLIENT_EMAIL').replace('@', '%40')}",
            "universe_domain": "googleapis.com"
        }
        
        creds = service_account.Credentials.from_service_account_info(credentials_info)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        
        print(f"⏳ Tentative de chargement de {model_name}...")
        model = GenerativeModel(model_name)
        print(f"✅ Modèle {model_name} prêt.")

    except Exception as e:
        print("❌ ERREUR CRITIQUE INITIALISATION VERTEX :")
        print(e)
        # Fallback au cas où 2.5 Pro n'est pas dispo
        try:
            print("⚠️ Repli sur gemini-1.5-pro...")
            model_name = "gemini-1.5-pro"
            model = GenerativeModel(model_name)
        except:
            pass

init_vertex()

# Hook pour s'assurer que les headers CORS sont TOUJOURS présents
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"!!! SERVER CRASH !!! : {str(e)}")
    print(traceback.format_exc())
    return jsonify({
        "success": False, 
        "error": str(e),
        "model": model_name,
        "trace": traceback.format_exc()
    }), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online", 
        "model": model_name, 
        "info": "CORS rules simplified for production"
    })

@app.route('/scan', methods=['POST'])
def scan_image():
    global model
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image field in request"}), 400

    if model is None:
         return jsonify({"success": False, "error": "Vertex AI initialization failed"}), 500

    file = request.files['image']
    
    try:
        img_bytes = file.read()
        
        mime_type = file.content_type
        if img_bytes.startswith(b'\x89PNG'): mime_type = 'image/png'
        elif img_bytes.startswith(b'\xff\xd8'): mime_type = 'image/jpeg'

        image_part = Part.from_data(data=img_bytes, mime_type=mime_type if mime_type else "image/jpeg")
        prompt = "1. Extract all text from this image, without any comments or explanations."

        print(f"📡 Calling Vertex AI ({model_name})...")
        
        response = model.generate_content(
            [image_part, prompt],
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1
            }
        )

        if not response.text:
            return jsonify({"success": False, "error": "AI returned empty response"}), 500

        return jsonify({"success": True, "text": response.text.strip()})

    except Exception as e:
        print(f"❌ OCR ERROR: {str(e)}")
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)