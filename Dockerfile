FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download BLIP model ONCE during build
RUN python -c "\
from transformers import BlipProcessor, BlipForConditionalGeneration;\
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='blip-model');\
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='blip-model')"

# Copy the rest of the code
COPY . .

EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
