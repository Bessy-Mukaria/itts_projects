from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import torch
import os
import uuid
import time

app = FastAPI()

# Serve static files like generated audio
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model variables
processor = None
model = None

@app.on_event("startup")
async def load_model():
    global processor, model

    print("🔄 Loading BLIP model from local folder...")
    processor = BlipProcessor.from_pretrained("blip-model")
    model = BlipForConditionalGeneration.from_pretrained("blip-model")
    print("✅ Model and processor loaded successfully.")

@app.get("/")
async def home():
    return FileResponse("templates/index.html")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image = Image.open(file.file).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    audio_filename = f"output_{uuid.uuid4().hex}.mp3"
    audio_path = f"static/{audio_filename}"
    gTTS(text=caption, lang='en').save(audio_path)

    elapsed_time = time.time() - start_time

    return JSONResponse({
        "caption": caption,
        "audio_path": f"/static/{audio_filename}",
        "processing_time": f"{elapsed_time:.2f} seconds"
    })
