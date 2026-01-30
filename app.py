import os
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from google import genai
from google.genai import types
import traceback
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# CORS : Très permissif pour éviter les blocages
CORS(app, resources={r"/*": {"origins": "*"}})

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "global" # Reverted to global as it worked locally

def get_client():
    try:
        print(f"🔧 Starting Vertex AI Config for Project: {PROJECT_ID}")
        
        # 1. Nettoyage de la clé privée (Source fréquente d'erreurs)
        raw_key = os.getenv("GOOGLE_PRIVATE_KEY", "")
        if not raw_key:
            print("❌ ERREUR: GOOGLE_PRIVATE_KEY est vide sur le serveur !")
            return None
            
        pk = raw_key.replace('\\n', '\n').strip()
        if pk.startswith('"') and pk.endswith('"'): pk = pk[1:-1]
        
        # Debug (Sécurisé : on n'affiche que le début)
        print(f"🔑 Clé chargée (début): {pk[:15]}...")
        
        client_email = os.getenv("GOOGLE_CLIENT_EMAIL")
        print(f"📧 Service Account Email: {client_email}")

        # 2. Création du dictionnaire de credentials
        credentials_info = {
            "type": "service_account",
            "project_id": PROJECT_ID,
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": pk,
            "client_email": client_email,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GOOGLE_CLIENT_EMAIL', '').replace('@', '%40')}",
            "universe_domain": "googleapis.com"
        }
        
        # 3. Injection dans un fichier temporaire (Obligatoire pour le SDK google-genai)
        temp_creds = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(credentials_info, temp_creds)
        temp_creds.close()
        
        # 4. Définition de la variable d'environnement ADC
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds.name
        
        print(f"✅ Fichier credentials généré: {temp_creds.name}")
        
        # 5. Création du client
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        return client
        return client
    except Exception as e:
        print(f"❌ Erreur Init Client Vertex: {str(e)}")
        # On log l'erreur complète pour la voir dans Render
        print(traceback.format_exc())
        return None

client = get_client()

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "online", 
        "mode": "vertex-ai",
        "client_ready": client is not None
    })

@app.errorhandler(500)
def internal_error(error):
    response = jsonify({
        "success": False,
        "error": "Internal Server Error",
        "details": str(error)
    })
    response.status_code = 500
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # Now you're handling non-HTTP exceptions only
    print(f"❌ Unhandled Exception: {str(e)}")
    print(traceback.format_exc())
    response = jsonify({
        "success": False,
        "error": "Unhandled Exception",
        "details": str(e)
    })
    response.status_code = 500
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/scan', methods=['POST'])
def scan_image():
    global client
    
    # Tentative de ré-init si perdu
    if not client:
        client = get_client()
        if not client:
            return jsonify({"success": False, "error": "Server Credential Error. Check Render Logs."}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400

    file = request.files['image']
    img_bytes = file.read()
    mime = file.content_type or "image/jpeg"
    
    # Modèle à utiliser
    target_model = "gemini-2.5-flash"
    
    try:
        print(f"🚀 Scan avec {target_model} ({LOCATION})...")
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        
        response = client.models.generate_content(
            model=target_model,
            contents=[image_part, "1. Extract all text from this image, without any comments or explanations."],
            config=types.GenerateContentConfig(temperature=0)
        )
        
        return jsonify({"success": True, "text": response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erreur Gemini 2.5 Flash: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            "success": False, 
            "error": f"Erreur OCR: {error_msg}",
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)