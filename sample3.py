from fastapi import FastAPI, File, UploadFile, Form
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
import base64
import librosa
from datetime import datetime
# from TTS.api import TTS
# import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# tts.to(device)

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
from collections import Counter
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
from checking import UnifiedDeepfakeDetector

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
    with open(reencoded_filename, "rb") as audio_file:
        audio_data = audio_file.read()

    # audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    os.remove(reencoded_filename)
    return JSONResponse(content={
        "prediction": bool(prediction),
        "entropy": float(entropy),
    })
    
    
@app.post("/upload_audio")
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
    detector = UnifiedDeepfakeDetector()
    print(reencoded_filename)
    result = detector.analyze_audio_rf(reencoded_filename, model_choice="all")
    prediction, entropy = classify_audio(new_features)
    with open(reencoded_filename, "rb") as audio_file:
        audio_data = audio_file.read()
    result.append("FAKE" if float(entropy) < 150 else "REAL")
    print(result)
    r_normalized = [x.upper() for x in result]
    counter = Counter(r_normalized)

    # Find the most common element
    most_common_element, _ = counter.most_common(1)[0]

    print(f"The most frequent element is: {most_common_element}") 
    

    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print(f"Audio Data Length: {len(audio_data)}")

    os.remove(reencoded_filename)
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": most_common_element.upper(),
        "entropy": float(entropy),
        "audio": audio_base64,
        "content_type": "audio/wav"
    })

# @app.post("/upload_deepfake")
# async def upload_file(file: UploadFile = File(...), text: str = Form(...)):
#     print(f"Received file: {file.filename}")

#     original_filename = file.filename.rsplit('.', 1)[0]
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     wav_filename = os.path.join(SAVE_DIR, f"{timestamp}.wav")
#     reencoded_filename = os.path.join(SAVE_DIR, f"{timestamp}_reencoded.wav")

#     os.makedirs(SAVE_DIR, exist_ok=True)
#     with open(wav_filename, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     reencode_audio(wav_filename, reencoded_filename)
#     os.remove(wav_filename)
#     print(f"File successfully re-encoded as: {reencoded_filename}")
#     try:
#         audio, sr = librosa.load(reencoded_filename, sr=None)  
#         print("Loaded successfully with librosa")
#     except Exception as e:
#         print(f"Error loading re-encoded file: {e}")
#     # tortoise(reencoded_filename, text)
    
#     # with open("/audio/sample.wav", "rb") as audio_file:
#     #     audio_data = audio_file.read()
#     with open(reencoded_filename, "rb") as audio_file:
#         audio_data = audio_file.read()
    

#     audio_base64 = base64.b64encode(audio_data).decode('utf-8')
#     print(f"Audio Data Length: {len(audio_data)}")
#     print(text)

#     os.remove(reencoded_filename)
#     # os.remove("/audio/sample.wav")
    
#     return JSONResponse(content={
#         "filename": file.filename,
#         "audio": audio_base64,
#         "content_type": "audio/wav"
#     })





# text_to_synthesize = "John Wick follows the story of a retired hitman, John Wick, portrayed by Keanu Reeves, who is grieving the loss of his wife, Helen. After receiving a puppy named Daisy as a final gift from her to help him cope, John's life takes a dark turn when a group of Russian gangsters led by Iosef Tarasov breaks into his home, steals his car, and brutally kills Daisy. This act of violence forces John back into his former life as a legendary assassin, igniting a relentless quest for vengeance against Iosef and his father, Viggo Tarasov, who is a powerful crime lord. As John battles through the criminal underworld, he faces numerous adversaries while seeking retribution for the loss of his beloved dog and the peace he once had. The film explores themes of grief, revenge, and the consequences of one's past actions, establishing John Wick as a formidable figure in the world of assassins."
# def tortoise(file, text):
#     output_path = "/audio/output.wav"
#     speaker_wav_path = file

#     tts.tts_to_file(
#         text=text,
#         file_path=output_path,
#         speaker_wav=speaker_wav_path,
#         language="en"
#     )

#     print(f"Audio has been saved to {output_path}")
    
    
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
    print(entropy)

    if  entropy > 150:
        return True, entropy
    else:
        return False, entropy
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)