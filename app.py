import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- CONFIGURATION VERTEX AI ---
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "us-central1"

print(f"--- Initialisation Vertex AI ({PROJECT_ID}) ---")

model = None

try:
    private_key = os.getenv("GOOGLE_PRIVATE_KEY")
    if private_key:
        private_key = private_key.replace('\\n', '\n')

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
    
    # PASSAGE À GEMINI 2.5 PRO pour une précision maximale
    model_name = "gemini-2.5-pro" 
    print(f"⏳ Chargement du modèle {model_name}...")
    model = GenerativeModel(model_name)
    print("✅ Modèle Vertex AI chargé avec succès.")

except Exception as e:
    print("❌ ERREUR INITIALISATION VERTEX :")
    print(e)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "model": "gemini-2.5-pro"})

@app.route('/scan', methods=['POST'])
def scan_image():
    global model
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image found"}), 400

    if model is None:
        return jsonify({"success": False, "error": "AI model not loaded"}), 500

    file = request.files['image']
    
    try:
        img_bytes = file.read()
        
        # Détection du MIME type réel
        mime_type = file.content_type
        if img_bytes.startswith(b'\x89PNG'): mime_type = 'image/png'
        elif img_bytes.startswith(b'\xff\xd8'): mime_type = 'image/jpeg'

        image_part = Part.from_data(data=img_bytes, mime_type=mime_type if mime_type else "image/jpeg")

        # SIMPLE HIGH-ACCURACY PROMPT
        prompt = "1. Extract all text from this image, without any comments or explanations."

        print(f"🚀 OCR avec Gemini 2.5 Pro...")
        
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1
        }

        from vertexai.generative_models import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Image en premier pour une meilleure focalisation visuelle
        response = model.generate_content(
            [image_part, prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if not response.text:
            return jsonify({"success": False, "error": "Empty response from AI"}), 500

        return jsonify({"success": True, "text": response.text.strip()})

    except Exception as e:
        print(f"❌ ERREUR OCR : {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)