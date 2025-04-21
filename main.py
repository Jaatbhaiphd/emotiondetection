from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import librosa
import numpy as np
import pickle
from app.utils import predict_emotion
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    emotion = predict_emotion("temp.wav")
    os.remove("temp.wav")
    return {"emotion": emotion}