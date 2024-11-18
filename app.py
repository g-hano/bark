from flask import Flask, request, jsonify, send_file, render_template
import logging
import numpy as np
import nltk
import torch
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import os

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUDIO_OUTPUT_DIR = "static/audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 24000  # Standard sample rate for Bark output
SILENCE_DURATION = 0.25  # quarter-second silence between sentences

nltk.download("punkt")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {device}")

try:
    models = {
        "suno/bark": BarkModel.from_pretrained("suno/bark").to(device),
        "suno/bark-small": BarkModel.from_pretrained("suno/bark-small").to(device)
    }
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    models = {}

# Voice presets
all_voice_presets = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
    "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6",
    "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9",
    "v2/tur_speaker_0", "v2/tur_speaker_1", "v2/tur_speaker_2", "v2/tur_speaker_3",
    "v2/tur_speaker_4", "v2/tur_speaker_5", "v2/tur_speaker_6",
    "v2/tur_speaker_7", "v2/tur_speaker_8", "v2/tur_speaker_9",
    "v2/de_speaker_0", "v2/de_speaker_1", "v2/de_speaker_2", "v2/de_speaker_3",
    "v2/de_speaker_4", "v2/de_speaker_5", "v2/de_speaker_6",
    "v2/de_speaker_7", "v2/de_speaker_8", "v2/de_speaker_9",
    "v2/fr_speaker_0", "v2/fr_speaker_1", "v2/fr_speaker_2", "v2/fr_speaker_3",
    "v2/fr_speaker_4", "v2/fr_speaker_5", "v2/fr_speaker_6",
    "v2/fr_speaker_7", "v2/fr_speaker_8", "v2/fr_speaker_9",
    "v2/it_speaker_0", "v2/it_speaker_1", "v2/it_speaker_2", "v2/it_speaker_3",
    "v2/it_speaker_4", "v2/it_speaker_5", "v2/it_speaker_6",
    "v2/it_speaker_7", "v2/it_speaker_8", "v2/it_speaker_9",
    "v2/zh_speaker_0", "v2/zh_speaker_1", "v2/zh_speaker_2", "v2/zh_speaker_3",
    "v2/zh_speaker_4", "v2/zh_speaker_5", "v2/zh_speaker_6",
    "v2/zh_speaker_7", "v2/zh_speaker_8", "v2/zh_speaker_9"
]

def generate_speech(text, model_name, voice_preset):
    # Generate speech audio from text using the selected model and voice preset.
    try:
        print(f"{text=} | {model_name=}")
        model = models[model_name].to(device)
        print(f"{model.device=}")
        processor = AutoProcessor.from_pretrained(model_name)
        inputs = processor(text, voice_preset=voice_preset)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
           audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        
        if audio_array.size == 0:
           print("Generated Audio is Empty")
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
        return audio_array
    except Exception as e:
        logging.error(f"Error in speech generation: {e}")
        raise

@app.route("/", methods=["GET"])
def index():
    # Render the main page.
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_voice():
    # Generate voice from the input text and return the audio file path.
    try:
        text = request.form.get("text")
        model_name = request.form.get("model_name")
        voice_preset = request.form.get("voice_preset")

        if not text or not model_name or not voice_preset:
            raise ValueError("Missing required parameters: text, model_name, or voice_preset")

        audio_data = generate_speech(text, model_name, voice_preset)

        # Save the audio file
        output_file = f"{AUDIO_OUTPUT_DIR}/generated_voice.wav"
        write(output_file, SAMPLE_RATE, audio_data)
        logging.info(f"Audio file saved at {output_file}")

        return jsonify({"file_path": f"/{output_file}"})
    except Exception as e:
        logging.error(f"Error generating voice: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/audio", methods=["GET"])
def get_audio():
    # Serve the generated audio file.
    try:
        file_path = f"{AUDIO_OUTPUT_DIR}/generated_voice.wav"
        if not os.path.exists(file_path):
            raise FileNotFoundError("Audio file not found")
        return send_file(file_path, mimetype="audio/wav")
    except Exception as e:
        logging.error(f"Error retrieving audio file: {e}")
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

