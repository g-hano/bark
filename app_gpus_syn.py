from flask import Flask, request, jsonify, send_file, render_template
import logging
import numpy as np
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import os
from typing import List, Tuple, Dict
import re
from queue import Queue
import threading
import time

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

AUDIO_OUTPUT_DIR = "static/audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.25

nltk.download("punkt", quiet=True)

# Get number of available GPUs
num_gpus = torch.cuda.device_count()
logging.info(f"Number of available GPUs: {num_gpus}")

if num_gpus == 0:
    device = torch.device("cpu")
    devices = [device]
else:
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Dict[torch.device, BarkModel]] = {}
        self.processors: Dict[str, AutoProcessor] = {}
        self.load_models()

    def load_models(self):
        """Load models on all available GPUs."""
        try:
            for model_name in ["suno/bark", "suno/bark-small"]:
                self.models[model_name] = {}
                self.processors[model_name] = AutoProcessor.from_pretrained(model_name)
                for device in devices:
                    model = BarkModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
                    model.eval()  # Set model to evaluation mode
                    self.models[model_name][device] = model
            logging.info("Models loaded successfully on all devices")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def get_model(self, model_name: str, device: torch.device) -> BarkModel:
        return self.models[model_name][device]

    def get_processor(self, model_name: str) -> AutoProcessor:
        return self.processors[model_name]

# Initialize model manager
model_start = time.time()
model_manager = ModelManager()
model_end = time.time()
print(f"Models loaded: {model_end - model_start}")

def has_sentence_endings(text: str) -> bool:
    """
    Check if text contains sentence ending punctuation marks.
    """
    sentence_endings = ['!', '?', '.', '...']
    return any(ending in text for ending in sentence_endings)

def split_by_sentences(text: str) -> List[str]:
    """
    Split text into sentences based on punctuation marks.
    """
    # First handle ellipsis to avoid wrong splits
    text = text.replace('...', '<ELLIPSIS>')
    
    # Split by sentence endings
    sentences = []
    for sentence in re.split('[!?.]', text):
        sentence = sentence.strip()
        if sentence:
            # Restore ellipsis
            sentence = sentence.replace('<ELLIPSIS>', '...')
            sentences.append(sentence)
    
    return sentences

def split_by_words(text: str, num_gpus: int) -> List[str]:
    """
    Split text into balanced chunks based on word count and available GPUs.
    """
    words = text.split()
    total_words = len(words)
    MIN_WORDS_PER_GPU = 10
    if total_words < MIN_WORDS_PER_GPU * 2:  # At least 20 words to split between 2 GPUs
        logging.info(f"Text too short ({total_words} words), using single GPU")
        return [text]

    needed_gpus = min(
        num_gpus,
        max(1, total_words // MIN_WORDS_PER_GPU)
    )
    
    # If we only need one GPU, return the whole text
    if needed_gpus == 1:
        return [text]
    
    words_per_gpu = total_words // needed_gpus
    remainder = total_words % needed_gpus
    chunks = []
    start_idx = 0
    
    for i in range(num_gpus):
        # Add one extra word to some chunks if there's a remainder
        chunk_size = words_per_gpu + (1 if i < remainder else 0)
        if chunk_size > 0 and start_idx < len(words):
            chunk = ' '.join(words[start_idx:start_idx + chunk_size])
            chunks.append(chunk)
            start_idx += chunk_size
    
    return [chunk for chunk in chunks if chunk.strip()]

def optimize_chunk_distribution(text: str) -> List[str]:
    """
    Distribute text across available GPUs based on content.
    """
    preprocess_start = time.time()
    if not text.strip():
        logging.warning("Empty text received")
        return []
    
    if has_sentence_endings(text):
        logging.info("Text contains sentence endings, splitting by sentences")
        chunks = split_by_sentences(text)
        logging.info(f"Split into {len(chunks)} sentences")
    else:
        logging.info("Text doesn't contain sentence endings, splitting by word count")
        chunks = split_by_words(text, num_gpus)
        logging.info(f"Split into {len(chunks)} word-based chunks for {num_gpus} GPUs")
    
    # Log distribution details
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        logging.info(f"Chunk {i + 1}: {word_count} words")
    preprocess_end = time.time()
    print(f"Preprocess Time: {preprocess_end - preprocess_start}")
    return chunks

def generate_speech_chunk(args: Tuple[str, str, str, torch.device, int]) -> Tuple[int, np.ndarray]:
    """Generate speech for a single chunk of text on specified device."""
    text, model_name, voice_preset, device, chunk_idx = args
    try:
        model = model_manager.get_model(model_name, device)
        processor = model_manager.get_processor(model_name)
        
        inputs = processor(text, voice_preset=voice_preset)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = model.generate(**inputs)
            
        audio_array = audio_array.cpu().numpy().squeeze()
        if audio_array.size == 0:
            logging.warning(f"Generated audio is empty for chunk: {text}")
            return chunk_idx, np.array([])
            
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return chunk_idx, audio_array
    except Exception as e:
        logging.error(f"Error in chunk speech generation: {e}")
        raise

def add_silence(audio: np.ndarray, duration: float = SILENCE_DURATION) -> np.ndarray:
    """Add silence of specified duration to the audio."""
    silence_samples = int(SAMPLE_RATE * duration)
    silence = np.zeros(silence_samples)
    return np.concatenate([audio, silence])

def generate_speech_parallel(text: str, model_name: str, voice_preset: str) -> np.ndarray:
    """Generate speech in parallel using available GPUs efficiently."""
    chunks = optimize_chunk_distribution(text)
    num_chunks = len(chunks)
    
    if num_chunks == 0:
        logging.warning("No text chunks to process")
        return np.array([])
    
    # Calculate actual number of GPUs needed
    gpus_needed = min(num_chunks, num_gpus)
    devices_to_use = devices[:gpus_needed]
    
    logging.info(f"Using {gpus_needed} GPUs for {num_chunks} chunks")
    
    # Prepare arguments for parallel processing
    chunk_args = []
    for i, chunk in enumerate(chunks):
        device_idx = i % len(devices_to_use)
        chunk_args.append((chunk, model_name, voice_preset, devices_to_use[device_idx], i))
    
    # Process chunks in parallel
    audio_chunks = [None] * num_chunks
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=gpus_needed) as executor:
        future_to_idx = {executor.submit(generate_speech_chunk, args): args[4] 
                        for args in chunk_args}
        
        for future in as_completed(future_to_idx):
            try:
                chunk_idx, audio_chunk = future.result()
                if audio_chunk.size > 0:
                    if chunk_idx < num_chunks - 1:
                        audio_chunk = add_silence(audio_chunk)
                    audio_chunks[chunk_idx] = audio_chunk
            except Exception as e:
                logging.error(f"Error processing chunk {future_to_idx[future]}: {str(e)}")
    
    # Remove None values and concatenate
    audio_chunks = [chunk for chunk in audio_chunks if chunk is not None]
    final_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
    
    processing_time = time.time() - start_time
    logging.info(f"Speech generation completed in {processing_time:.2f} seconds")
    
    return final_audio

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_voice():
    try:
        text = request.form.get("text")
        model_name = request.form.get("model_name", "suno/bark")  # Default to bark model
        voice_preset = request.form.get("voice_preset", "v2/en_speaker_6")  # Default voice

        if not text:
            raise ValueError("Missing required parameter: text")

        # Generate unique filename using timestamp
        timestamp = int(time.time())
        output_filename = f"generated_voice_{timestamp}.wav"
        output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

        # Generate speech using parallel processing
        audio_data = generate_speech_parallel(text, model_name, voice_preset)
        
        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save the audio file
        save_start = time.time()
        write(output_path, SAMPLE_RATE, audio_data)
        save_end = time.time()
        print(f"Voice written: {save_end - save_start}")
        logging.info(f"Audio file saved at {output_path}")

        return jsonify({
            "success": True,
            "file_path": f"/static/audio/{output_filename}",
            "message": "Audio generated successfully"
        })

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error generating voice: {error_msg}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify service status."""
    return jsonify({
        "status": "healthy",
        "gpu_count": num_gpus,
        "timestamp": time.time()
    })

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # Add startup logging
    logging.info(f"Starting Flask application with {num_gpus} GPUs")
    logging.info(f"Output directory: {AUDIO_OUTPUT_DIR}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

