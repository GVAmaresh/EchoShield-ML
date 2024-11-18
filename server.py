from fastapi import FastAPI, WebSocket
import io
import base64
import torchaudio
from models.wav2vec import extract_features, classify_audio

app = FastAPI()

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        base64_audio = data['data']
        audio_data = base64.b64decode(base64_audio)
        print(f"Decoded audio data length: {len(audio_data)}")  
        
        if len(audio_data) == 0:
            await websocket.send_json({"error": "Received empty audio data"})
            continue
        audio_bytes = io.BytesIO(audio_data)
        try:
            prediction, entropy = process_audio(audio_bytes)
            print(prediction, entropy)
            await websocket.send_json({"prediction": prediction, "entropy": entropy})
        except Exception as e:
            print(f"Error processing audio: {e}")
            await websocket.send_json({"error": str(e)})

def process_audio(audio_bytes):
    try:
        waveform, sample_rate = torchaudio.load(audio_bytes, format="wav")
        data = waveform.squeeze().numpy()
        print(f"Audio data loaded: {data.shape}, Sample rate: {sample_rate}")

        features = extract_features(data, sample_rate)
        prediction, entropy = classify_audio(features)
        return prediction, entropy
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"error": str(e)}, None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

