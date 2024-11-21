from flask import Flask, request, jsonify, send_file, render_template
import logging
import numpy as np
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import os
from typing import List, Tuple, Dict, Optional
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

num_gpus = torch.cuda.device_count()
logging.info(f"Number of available GPUs: {num_gpus}")

if num_gpus == 0:
    device = torch.device("cpu")
    devices = [device]
else:
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

class ModelManager:
    def __init__(self):
        self.current_model_name: Optional[str] = None
        self.models: Dict[torch.device, BarkModel] = {}
        self.processor: Optional[AutoProcessor] = None
        self.model_lock = threading.Lock()

    def load_model(self, model_name: str) -> float:
        """
        Load model and return the time taken to load.
        """
        start_time = time.time()
        
        with self.model_lock:
            # If same model is already loaded, return immediately
            if self.current_model_name == model_name:
                return 0.0
            
            # Unload previous model if exists
            self.unload_current_model()
            
            try:
                # Load processor
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Load model on all available devices
                for device in devices:
                    model = BarkModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16
                    ).to(device)
                    model.eval()
                    self.models[device] = model
                
                self.current_model_name = model_name
                load_time = time.time() - start_time
                logging.info(f"Model {model_name} loaded successfully in {load_time:.2f} seconds")
                return load_time
                
            except Exception as e:
                self.unload_current_model()
                logging.error(f"Error loading model {model_name}: {e}")
                raise

    def unload_current_model(self):
        """
        Unload the current model and clear GPU memory.
        """
        if self.current_model_name:
            try:
                for device, model in self.models.items():
                    del model
                self.models.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.processor = None
                self.current_model_name = None
                logging.info("Previous model unloaded successfully")
            except Exception as e:
                logging.error(f"Error unloading model: {e}")

    def get_model(self, device: torch.device) -> BarkModel:
        return self.models[device]

    def get_processor(self) -> AutoProcessor:
        return self.processor

model_manager = ModelManager()

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
    text = text.replace('...', '<ELLIPSIS>')
    
    sentences = []
    for sentence in re.split('[!?.]', text):
        sentence = sentence.strip()
        if sentence:
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
    if total_words < MIN_WORDS_PER_GPU * 2:
        logging.info(f"Text too short ({total_words} words), using single GPU")
        return [text]

    needed_gpus = min(
        num_gpus,
        max(1, total_words // MIN_WORDS_PER_GPU)
    )
    
    if needed_gpus == 1:
        return [text]
    
    words_per_gpu = total_words // needed_gpus
    remainder = total_words % needed_gpus
    chunks = []
    start_idx = 0
    
    for i in range(num_gpus):
        chunk_size = words_per_gpu + (1 if i < remainder else 0)
        if chunk_size > 0 and start_idx < len(words):
            chunk = ' '.join(words[start_idx:start_idx + chunk_size])
            chunks.append(chunk)
            start_idx += chunk_size
    
    return [chunk for chunk in chunks if chunk.strip()]

def optimize_chunk_distribution(text: str) -> Tuple[List[str], float]:
    """
    Distribute text across available GPUs based on content.
    Returns chunks and time taken for preprocessing.
    """
    start_time = time.time()
    
    if not text.strip():
        logging.warning("Empty text received")
        return [], 0.0
    
    if has_sentence_endings(text):
        logging.info("Text contains sentence endings, splitting by sentences")
        chunks = split_by_sentences(text)
        logging.info(f"Split into {len(chunks)} sentences")
    else:
        logging.info("Text doesn't contain sentence endings, splitting by word count")
        chunks = split_by_words(text, num_gpus)
        logging.info(f"Split into {len(chunks)} word-based chunks for {num_gpus} GPUs")
    
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        logging.info(f"Chunk {i + 1}: {word_count} words")
    
    preprocess_time = time.time() - start_time
    logging.info(f"Text preprocessing completed in {preprocess_time:.2f} seconds")
    
    return chunks, preprocess_time

def generate_speech_chunk(args: Tuple[str, str, torch.device, int]) -> Tuple[int, np.ndarray, float]:
    """Generate speech for a single chunk of text on specified device."""
    text, voice_preset, device, chunk_idx = args
    start_time = time.time()
    
    try:
        model = model_manager.get_model(device)
        processor = model_manager.get_processor()
        
        inputs = processor(text, voice_preset=voice_preset)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = model.generate(**inputs)
            
        audio_array = audio_array.cpu().numpy().squeeze()
        if audio_array.size == 0:
            logging.warning(f"Generated audio is empty for chunk: {text}")
            return chunk_idx, np.array([]), 0.0
            
        audio_array = np.clip(audio_array, -1.0, 1.0)
        generation_time = time.time() - start_time
        
        return chunk_idx, audio_array, generation_time
        
    except Exception as e:
        logging.error(f"Error in chunk speech generation: {e}")
        raise

def add_silence(audio: np.ndarray, duration: float = SILENCE_DURATION) -> np.ndarray:
    """Add silence of specified duration to the audio."""
    silence_samples = int(SAMPLE_RATE * duration)
    silence = np.zeros(silence_samples)
    return np.concatenate([audio, silence])

def generate_speech_parallel(text: str, voice_preset: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate speech in parallel using available GPUs efficiently."""
    timing_info = {}
    
    # Text preprocessing
    chunks, timing_info['preprocessing_time'] = optimize_chunk_distribution(text)
    num_chunks = len(chunks)
    
    if num_chunks == 0:
        logging.warning("No text chunks to process")
        return np.array([]), timing_info

    gpus_needed = min(num_chunks, num_gpus)
    devices_to_use = devices[:gpus_needed]
    
    logging.info(f"Using {gpus_needed} GPUs for {num_chunks} chunks")
    
    chunk_args = []
    for i, chunk in enumerate(chunks):
        device_idx = i % len(devices_to_use)
        chunk_args.append((chunk, voice_preset, devices_to_use[device_idx], i))
    
    audio_chunks = [None] * num_chunks
    chunk_times = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=gpus_needed) as executor:
        future_to_idx = {executor.submit(generate_speech_chunk, args): args[3] 
                        for args in chunk_args}
        
        for future in as_completed(future_to_idx):
            try:
                chunk_idx, audio_chunk, chunk_time = future.result()
                chunk_times.append(chunk_time)
                
                if audio_chunk.size > 0:
                    if chunk_idx < num_chunks - 1:
                        audio_chunk = add_silence(audio_chunk)
                    audio_chunks[chunk_idx] = audio_chunk
            except Exception as e:
                logging.error(f"Error processing chunk {future_to_idx[future]}: {str(e)}")
    
    audio_chunks = [chunk for chunk in audio_chunks if chunk is not None]
    final_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
    
    timing_info['total_generation_time'] = time.time() - start_time
    timing_info['average_chunk_time'] = sum(chunk_times) / len(chunk_times) if chunk_times else 0
    timing_info['max_chunk_time'] = max(chunk_times) if chunk_times else 0
    
    logging.info(f"Speech generation completed in {timing_info['total_generation_time']:.2f} seconds")
    
    return final_audio, timing_info

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_voice():
    try:
        text = request.form.get("text")
        model_name = request.form.get("model_name", "suno/bark")
        voice_preset = request.form.get("voice_preset", "v2/en_speaker_0")

        if not text:
            raise ValueError("Missing required parameter: text")

        # Load model if needed and measure time
        model_load_time = model_manager.load_model(model_name)

        # Generate unique filename using timestamp
        timestamp = int(time.time())
        output_filename = f"generated_voice_{timestamp}.wav"
        output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

        # Generate speech using parallel processing
        audio_data, timing_info = generate_speech_parallel(text, voice_preset)
        
        # Add model loading time to timing info
        timing_info['model_load_time'] = model_load_time
        
        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save the audio file
        write(output_path, SAMPLE_RATE, audio_data)
        logging.info(f"Audio file saved at {output_path}")

        return jsonify({
            "success": True,
            "file_path": f"/static/audio/{output_filename}",
            "message": "Audio generated successfully",
            "timing_info": timing_info
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
        "current_model": model_manager.current_model_name,
        "timestamp": time.time()
    })

if __name__ == "__main__":
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    logging.info(f"Starting Flask application with {num_gpus} GPUs")
    logging.info(f"Output directory: {AUDIO_OUTPUT_DIR}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
