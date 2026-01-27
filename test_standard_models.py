
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = "us-central1"

print(f"--- Listing Models for Project: {PROJECT_ID} in {LOCATION} ---")

try:
    credentials_info = {
        "type": "service_account",
        "project_id": PROJECT_ID,
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n'),
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

    # Use the gapic client or simply try a broader list of known stable models
    # There is no simple list_models in the high-level SDK that works easily without Model Garden permissions which might be different.
    # But we can try the standard non-versioned aliases which are often best.
    
    candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
    ]

    for model_name in candidates:
        try:
            print(f"Testing {model_name}...")
            model = GenerativeModel(model_name)
            response = model.generate_content("Hi")
            print(f"✅ SUCCESS: {model_name}")
        except Exception as e:
            print(f"❌ FAILED: {model_name} - {e}")

except Exception as e:
    print(e)
