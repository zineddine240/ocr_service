import os
import base64
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
# Pour Gemini 3 sur Vertex AI, la doc impose "global"
LOCATION = "global"

# Configuration du client avec le nouveau SDK Google Gen AI (en mode Vertex)
def get_client():
    try:
        pk = os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n').strip()
        if pk.startswith('"') and pk.endswith('"'): pk = pk[1:-1]
        
        # On passe les credentials via l'environnement pour le SDK
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        
        # Le nouveau SDK peut utiliser une clé API ou les credentials par défaut
        # Pour Vertex AI avec Service Account, on initialise avec vertexai=True
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        return client
    except Exception as e:
        print(f"Erreur Client: {str(e)}")
        return None

client = get_client()

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "online", "platform": "vertex-ai", "model": "gemini-3-flash-preview"})

@app.route('/scan', methods=['POST'])
def scan_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400

    file = request.files['image']
    img_bytes = file.read()
    mime = file.content_type or "image/jpeg"
    
    # Préparation de l'image pour le nouveau SDK
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
    prompt = "1. Extract all text from this image, without any comments or explanations."

    try:
        print("🚀 Appel Vertex AI - Gemini 3 Flash Preview (Location: Global)")
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8192
            )
        )
        
        return jsonify({"success": True, "text": response.text})

    except Exception as e:
        print(f"⚠️ Échec Gemini 3: {str(e)}")
        # Tentative de repli automatique sur 1.5 en cas d'erreur (ex: région non supportée)
        try:
            print("🔄 Tentative de repli sur Gemini 1.5 Flash...")
            # Pour 1.5, on repasse souvent par une localisation plus standard
            client_fallback = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")
            response = client_fallback.models.generate_content(
                model="gemini-1.5-flash",
                contents=[image_part, prompt]
            )
            return jsonify({"success": True, "text": response.text, "info": "Bascule auto sur 1.5 Flash"})
        except Exception as e2:
            return jsonify({"success": False, "error": f"Erreur critique: {str(e2)}", "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)