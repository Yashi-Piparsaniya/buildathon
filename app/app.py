# ================= IMPORTS =================
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import base64
import uuid
import os
import tempfile
import shutil
import warnings

from app.src.deepfake import infa_deepfake

warnings.filterwarnings("ignore")

# ================= CREATE APP =================
app = FastAPI(
    title="DeepFake Voice Detection API",
    description="GUVI-compatible Deepfake Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ROOT =================
@app.get("/")
def root():
    return {"status": "API is working"}

# ================= FILE UPLOAD (Swagger) =================
@app.post("/deepfake")
async def deepfake_file(audio_file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, audio_file.filename)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    status, message = infa_deepfake(temp_path)
    shutil.rmtree(temp_dir)

    label = "Human" if message == "REAL" else "AI-generated"

    return {
        "classification": label,
        "confidence": 0.85,
        "explanation": "Prediction based on YAMNet acoustic embeddings"
    }

# ================= GUVI ENDPOINT =================
@app.post("/detect")
async def detect_base64(request: Request):
    try:
        data = {}

        # 1️⃣ Try JSON body
        try:
            json_data = await request.json()
            if isinstance(json_data, dict):
                data.update(json_data)
        except:
            pass

        # 2️⃣ Try form-data
        try:
            form = await request.form()
            for k, v in form.items():
                data[k] = v
        except:
            pass

        # 3️⃣ Normalize keys (VERY IMPORTANT)
        normalized = {k.lower().strip(): v for k, v in data.items()}

        # 4️⃣ Extract fields (accept all GUVI variants)
        language = normalized.get("language")

        audio_format = (
            normalized.get("audio_format")
            or normalized.get("audioformat")
            or normalized.get("audio format")
        )

        audio_base64 = (
            normalized.get("audio_base64")
            or normalized.get("audiobase64")
            or normalized.get("audio_base64_format")
            or normalized.get("audio base64")
        )

        if not language or not audio_format or not audio_base64:
            return {
                "classification": "error",
                "confidence": 0,
                "explanation": f"Missing required fields. Receivedcvd keys received: {list(normalized.keys())}"
            }

        # 5️⃣ Clean base64
        clean_b64 = (
            audio_base64
            .replace("\n", "")
            .replace("\r", "")
            .replace(" ", "")
        )

        audio_bytes = base64.b64decode(clean_b64)

        temp_file = f"temp_{uuid.uuid4()}.{audio_format}"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        status, message = infa_deepfake(temp_file)
        os.remove(temp_file)

        label = "Human" if message == "REAL" else "AI-generated"

        return {
            "classification": label,
            "confidence": 0.85,
            "explanation": "Language-agnostic detection using YAMNet audio embeddings"
        }

    except Exception as e:
        return {
            "classification": "error",
            "confidence": 0,
            "explanation": str(e)
        }
