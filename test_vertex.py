import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import time

# Configuration
PROJECT_ID = "rational-lambda-485021-e9"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

print(f"--- Diagnosing Vertex AI for project: {PROJECT_ID} ---")

try:
    with open("log.txt", "w", encoding="utf-8") as f:
        # Testing specific EU regions since user is in Spain
        configs = [
            ("gemini-1.5-flash", "europe-southwest1"), # Madrid
            ("gemini-1.5-flash", "europe-west9"),      # Paris
            ("gemini-1.0-pro", "europe-southwest1"),
            ("gemini-1.0-pro", "europe-west9")
        ]

        success = False
        for model_name, loc in configs:
            try:
                f.write(f"\n--- Testing {model_name} in {loc} ---\n")
                vertexai.init(project=PROJECT_ID, location=loc)
                model = GenerativeModel(model_name)
                response = model.generate_content("Hello")
                f.write(f"SUCCESS! Response: {response.text}\n")
                f.write(f"VALID CONFIG FOUND: location='{loc}', model='{model_name}'\n")
                success = True
                break
            except Exception as e:
                f.write(f"Failed: {e}\n")

except Exception as outer_e:
    print(f"Critical error writing to log file: {outer_e}")
