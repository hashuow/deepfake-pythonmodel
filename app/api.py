from fastapi import FastAPI, UploadFile
import shutil
import os
from model_utils import predict_audio

app = FastAPI()

UPLOAD_FOLDER = "temp_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(audio: UploadFile):
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = predict_audio(file_path)
    os.remove(file_path)  # Delete file after prediction (optional)

    return result
