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
from PIL import Image
import io

load_dotenv()

app = Flask(__name__)
# CORS : Très permissif pour éviter les blocages
CORS(app, resources={r"/*": {"origins": "*"}})

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
# 'global' is recommended for Gemini 3 Preview models on Vertex AI
LOCATION = "global" 

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
    
    if not client:
        client = get_client()
        if not client:
            return jsonify({"success": False, "error": "Vertex AI Client not initialized"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        # 1. Image Receipt
        start_receive = time.time()
        file = request.files['image']
        img_bytes = file.read()
        file_size = len(img_bytes) / (1024 * 1024)
        mime = file.content_type or "image/jpeg"
        print(f"📥 [STEP 1] Received {file.filename} ({file_size:.2f} MB) - Time: {time.time() - start_receive:.2f}s")
        
        start_process = time.time()
        # 2. Optimization: Resize if image is too large (> 1.5MB)
        if file_size > 1.5:
            print(f"⚙️ [STEP 2] Optimizing large image...")
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((1600, 1600))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_bytes = img_byte_arr.getvalue()
            print(f"✅ Optimized to {len(img_bytes)/(1024*1024):.2f} MB - Time: {time.time() - start_process:.2f}s")
        else:
            print(f"⏩ [STEP 2] Skipping optimization (small image)")

        # 3. Gemini Call
        start_ai = time.time()
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        print(f"⏳ [STEP 3] Calling Gemini 3 Flash Preview ({LOCATION})...")
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                image_part, 
                "Extract all text from image. No comments."
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048
            )
        )
        ai_duration = time.time() - start_ai
        print(f"✅ [DONE] AI Response received in {ai_duration:.2f}s")
        
        total_duration = time.time() - start_receive
        return jsonify({
            "success": True, 
            "text": response.text.strip(),
            "debug": {
                "total_time": f"{total_duration:.2f}s",
                "ai_time": f"{ai_duration:.2f}s",
                "optimized": file_size > 1.5
            }
        })

    except Exception as e:
        print(f"❌ Gemini 3 Error: {str(e)}")
        return jsonify({
            "success": False, 
            "error": "OCR processing failed with gemini-3-flash-preview",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Use port 5000 as default locally to match Vite config
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)