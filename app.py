import os
import json
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import traceback
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "global" # Obligatoire pour Gemini 3 Flash Preview sur Vertex

def get_client():
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
        
        # SOLUTION : Créer un fichier JSON temporaire pour les credentials (ADC)
        # C'est la méthode la plus fiable pour le nouveau SDK sur Render
        temp_creds = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(credentials_info, temp_creds)
        temp_creds.close()
        
        # On dit au SDK où trouver les credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds.name
        
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        return client
    except Exception as e:
        print(f"Erreur Initialisation Client: {str(e)}")
        return None

client = get_client()

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "online", "model": "gemini-3-flash-preview", "ready": client is not None})

@app.route('/scan', methods=['POST'])
def scan_image():
    global client
    if not client:
        client = get_client()
        if not client:
            return jsonify({"success": False, "error": "Vertex AI initialization failed"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400

    file = request.files['image']
    img_bytes = file.read()
    mime = file.content_type or "image/jpeg"
    
    try:
        print("🚀 Scan via Gemini 3 Flash Preview (Vertex Global)")
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                image_part, 
                "1. Extract all text from this image, without any comments or explanations."
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8192
            )
        )
        
        return jsonify({"success": True, "text": response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Erreur: {error_msg}")
        
        # Repli auto sur 1.5 si erreur de région ou de modèle indisponible
        try:
            print("🔄 Tentative de repli sur Gemini 1.5 Flash (us-central1)...")
            client_fallback = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location="us-central1"
            )
            image_part_f = types.Part.from_bytes(data=img_bytes, mime_type=mime)
            response_f = client_fallback.models.generate_content(
                model="gemini-1.5-flash",
                contents=[image_part_f, "1. Extract all text from this image, without any comments or explanations."]
            )
            return jsonify({"success": True, "text": response_f.text, "info": "Bascule auto sur 1.5 Flash"})
        except Exception as e2:
            return jsonify({
                "success": False, 
                "error": f"Erreur critique: {str(e2)}", 
                "original_error": error_msg,
                "trace": traceback.format_exc()
            }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)