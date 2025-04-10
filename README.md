# 🧠 Visionary Voice AI

**Visionary Voice AI** is an AI-powered assistive web app that converts clothing images into speech — built for and inspired by accessibility needs of visually impaired users.

This tool leverages **BLIP** for intelligent image captioning and **gTTS** for audio narration, making visual content more inclusive and meaningful.

---

## 🌟 Features

- 🖼️ **Image Captioning**: Generates human-like captions using the BLIP model from Hugging Face Transformers.
- 🗣️ **Text-to-Speech**: Converts generated captions into speech using Google Text-to-Speech.
- 🎧 **Audio Playback**: Listen to the image description immediately within the app.
- 🧩 **Simple, Accessible UI**: Lightweight HTML interface that’s clean and assistive-friendly.

---

## 🚀 How It Works

1. Upload a clothing-related image via the interface.
2. The BLIP model interprets and describes the image.
3. The description is converted into an `.mp3` audio file using `gTTS`.
4. The result is both visible and audible to the user.

---

## 🛠️ Tech Stack

| Tool              | Purpose                            |
|------------------|-------------------------------------|
| `FastAPI`        | Backend API for image upload and inference |
| `transformers`   | BLIP model for generating captions  |
| `torch`          | Deep learning inference engine      |
| `gTTS`           | Converts text to spoken audio       |
| `Pillow`         | Image handling and resizing         |
| `HTML/CSS`       | Frontend interface with audio preview |

---

## 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/Bessy-Mukaria/itts_projects.git
cd itts_projects

# Create a virtual environment
python -m venv venv
# Activate it:
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --reload
```

Once running, open your browser and visit:  
👉 `http://127.0.0.1:8000`

---

## 📁 Project Structure

```
itts_projects/
├── app.py               # Main FastAPI application
├── requirements.txt     # Python dependencies
├── static/              # Stores generated .mp3 audio files
├── templates/           # Contains the HTML UI
│   └── index.html
├── README.md            # You're here!
└── .gitignore
```

---

## 🧠 Future Improvements

- 🌍 Add multilingual support for caption-to-speech output
- 🧹 Clean up or auto-expire old audio files
- 💡 Optional switch to Streamlit for richer interactivity
- 🚀 Deploy with GPU acceleration for faster inference

---

## 👩🏽‍💻 Author

**Bessy Mukaria**  
MSc Data Science & Analytics
📌 [LinkedIn Profile](https://www.linkedin.com/in/bessy-mukaria)

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

> _“Bringing vision to voice — making digital content accessible for all.”_ 🎧💡
