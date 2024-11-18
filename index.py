from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn
from datetime import datetime

app = FastAPI()

origins = [
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = "./audio"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")

    original_filename = file.filename.rsplit('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(SAVE_DIR, f"{timestamp}.wav")

    with open(wav_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"File saved to: {wav_filename}")

    return {"filename": file.filename, "saved_as": wav_filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)