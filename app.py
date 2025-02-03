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

# Initialize the FastAPI application
app = FastAPI()

# Serve static files (e.g., audio and other assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for the model and processor
processor = None
model = None

@app.on_event("startup")
async def load_model():
    """
    Load the BLIP model and processor once when the server starts.
    """
    global processor, model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")

@app.get("/")
async def home():
    """
    Serve the homepage (index.html).
    """
    return FileResponse("templates/index.html")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """
    Handle image upload, generate caption, convert it to audio, and return the results.
    """
    # Start timing
    start_time = time.time()

    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Read and process the image
    image = Image.open(file.file).convert("RGB")

    # Generate caption using the BLIP model
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Generate a unique file name for the audio file
    audio_file_name = f"output_{uuid.uuid4().hex}.mp3"
    audio_path = f"static/{audio_file_name}"

    # Convert the caption to speech
    tts = gTTS(text=caption, lang='en')
    tts.save(audio_path)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log processing time
    print(f"Processing time: {elapsed_time:.2f} seconds")

    # Return the caption, audio path, and processing time
    return JSONResponse(content={
        "caption": caption,
        "audio_path": f"/static/{audio_file_name}",
        "processing_time": f"{elapsed_time:.2f} seconds"
    })
