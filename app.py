# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from gtts import gTTS
# import torch
# import os
# import uuid
# import time
# import pickle
# import gdown

# # Initialize the FastAPI application
# app = FastAPI()

# # Serve static files (e.g., audio and other assets)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Global variables for the model and processor
# processor = None
# model = None

# @app.on_event("startup")
# async def load_model():
#     """
#     Load the BLIP model and processor once when the server starts.
#     """
#     global processor, model

#     pickle_file_path = "models/blip.pkl"  # Path to the pickle file

#     # Load the model and processor from the pickle file
#     with open(pickle_file_path, "rb") as f:
#         processor, model = pickle.load(f)

#     print("Model and processor loaded from pickle file.")
#     # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     print("Model and processor loaded successfully.")

# @app.get("/")
# async def home():
#     """
#     Serve the homepage (index.html).
#     """
#     return FileResponse("templates/index.html")

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     """
#     Handle image upload, generate caption, convert it to audio, and return the results.
#     """
#     # Start timing
#     start_time = time.time()

#     # Ensure the uploaded file is an image
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Uploaded file must be an image")

#     # Read and process the image
#     image = Image.open(file.file).convert("RGB")

#     # Generate caption using the BLIP model
#     inputs = processor(image, return_tensors="pt")
#     with torch.no_grad():
#         out = model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)

#     # Generate a unique file name for the audio file
#     audio_file_name = f"output_{uuid.uuid4().hex}.mp3"
#     audio_path = f"static/{audio_file_name}"

#     # Convert the caption to speech
#     tts = gTTS(text=caption, lang='en')
#     tts.save(audio_path)

#     # End timing
#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     # Log processing time
#     print(f"Processing time: {elapsed_time:.2f} seconds")

#     # Return the caption, audio path, and processing time
#     return JSONResponse(content={
#         "caption": caption,
#         "audio_path": f"/static/{audio_file_name}",
#         "processing_time": f"{elapsed_time:.2f} seconds"
#     })
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

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load processor & model globally (once on startup)
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global processor, model
    print("🔄 Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    print("✅ Model loaded and ready.")

@app.get("/")
async def home():
    return FileResponse("templates/index.html")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image = Image.open(file.file).convert("RGB")
    # image = image.resize((224, 224))  # Resize to match model input size
    if image.size[0] > 500 or image.size[1] > 500:
        image = image.resize((384, 384))

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    audio_filename = f"output_{uuid.uuid4().hex}.mp3"
    audio_path = f"static/{audio_filename}"
    gTTS(text=caption, lang='en').save(audio_path)

    elapsed_time = time.time() - start_time
    print(f"⏱️ Processing time: {elapsed_time:.2f} seconds")

    return JSONResponse({
        "caption": caption,
        "audio_path": f"/static/{audio_filename}",
        "processing_time": f"{elapsed_time:.2f} seconds"
    })
