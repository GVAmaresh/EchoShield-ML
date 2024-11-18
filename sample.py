from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torchaudio
import torch
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import skew, kurtosis, median_abs_deviation
import io
import shutil
import os
import uvicorn
import soundfile as sf
import librosa
from datetime import datetime

app = FastAPI()

origins = [
"*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

SAVE_DIR = "./audio"
os.makedirs(SAVE_DIR, exist_ok=True)

import wave

import os
import shutil
from datetime import datetime
import soundfile as sf
import librosa
import torch.nn.functional as F
import subprocess
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

SAVE_DIR = './audio' 
def reencode_audio(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path
    ]
    subprocess.run(command, check=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")

    original_filename = file.filename.rsplit('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(SAVE_DIR, f"{timestamp}.wav")
    reencoded_filename = os.path.join(SAVE_DIR, f"{timestamp}_reencoded.wav")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(wav_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reencode_audio(wav_filename, reencoded_filename)
    os.remove(wav_filename)
    print(f"File successfully re-encoded as: {reencoded_filename}")

    try:
        audio, sr = librosa.load(reencoded_filename, sr=None)  
        print("Loaded successfully with librosa")
    except Exception as e:
        print(f"Error loading re-encoded file: {e}")
    new_features = extract_features(reencoded_filename)
    prediction, entropy = classify_audio(new_features)
    os.remove(reencoded_filename)
    return JSONResponse(content={"filename": file.filename, "prediction": bool(prediction), "entropy": float(entropy)})


def extract_features(file_path):
    if os.path.exists(file_path):
        print(f"File successfully written: {file_path}")
    else:
        print("File writing failed.")
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)(waveform)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    pooled_features = []
    for f in features:
        if f.dim() == 3:
            f = f.permute(0, 2, 1)
            pooled_f = F.adaptive_avg_pool1d(f[0].unsqueeze(0), 1).squeeze(0)
            pooled_features.append(pooled_f)

    final_features = torch.cat(pooled_features, dim=0).numpy()
    final_features = (final_features - np.mean(final_features)) / (np.std(final_features) + 1e-10)

    return final_features

def additional_features(features):
    mad = median_abs_deviation(features)
    features_clipped = np.clip(features, 1e-10, None)
    entropy = -np.sum(features_clipped * np.log(features_clipped))
    return mad, entropy

def classify_audio(features):

    _, entropy = additional_features(features)

    if  entropy > 150:
        return True, entropy
    else:
        return False, entropy
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)