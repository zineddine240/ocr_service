import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, Part

app = Flask(__name__)
# Configuration CORS ultra-permissive pour Render et Vercel
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
    # Reconstitution des identifiants depuis les variables d'environnement
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
    
    # Utilisation de Gemini 2.5 Flash
    model_name = "gemini-2.5-flash" 
    
    print(f"⏳ Chargement du modèle {model_name}...")
    model = GenerativeModel(
        model_name,
        system_instruction=["Extract all text from image, whitout any comments or explanations"]
    )
    print("✅ Modèle Vertex AI chargé avec succès.")

except Exception as e:
    print("❌ ERREUR INITIALISATION VERTEX :")
    print(e)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "OCR Service is running!", 
        "status": "online", 
        "deployed_model": "gemini-2.5-flash",
        "last_update": "14:26"
    })

@app.route('/scan', methods=['POST'])
def scan_image():
    global model
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400

    if model is None:
        return jsonify({"success": False, "error": "Le modèle AI n'est pas chargé (Erreur serveur)"}), 500

    file = request.files['image']
    
    try:
        # Lecture de l'image
        img_bytes = file.read()
        
        # Préparation pour Vertex AI
        image_part = Part.from_data(
            data=img_bytes, 
            mime_type=file.content_type if file.content_type else "image/jpeg"
        )

        # Prompt
        prompt = "Extract all text from image correctly, without any additional informations"

        print("🚀 Envoi à Vertex AI...")
        
        # Configuration de précision
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.0,
        }

        # Configuration de sécurité pour éviter les blocages sur des documents légitimes
        from vertexai.generative_models import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Retry logic for 429 Resource Exhausted
        import time
        from google.api_core.exceptions import ResourceExhausted

        response = None
        for attempt in range(4): # Try 4 times (Initial + 3 retries)
            try:
                response = model.generate_content(
                    [prompt, image_part],
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                break
            except ResourceExhausted:
                if attempt == 3:
                    raise # Re-raise if last attempt
                wait_time = 2 ** attempt # 1s, 2s, 4s
                print(f"⚠️ Quota dépassé (429), nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                raise e

        final_text = response.text
        print("✅ Réponse reçue !")
        
        return jsonify({"success": True, "text": final_text})

    except Exception as e:
        print(f"❌ ERREUR PENDANT LE SCAN : {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)