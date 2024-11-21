from flask import Flask, request, jsonify, send_file, render_template, session
import logging
import numpy as np
import torch
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import os
from typing import List, Tuple, Dict
import threading
import time
import uuid
from queue import Queue
from dataclasses import dataclass
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

AUDIO_OUTPUT_DIR = "static/audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 24000
SESSION_TIMEOUT = 300  # 5 minutes in seconds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

num_gpus = torch.cuda.device_count()
logging.info(f"Number of available GPUs: {num_gpus}")

if num_gpus == 0:
    device = torch.device("cpu")
    devices = [device]
else:
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

@dataclass
class UserSession:
    user_id: str
    gpu_id: int
    last_activity: datetime
    
class GPUManager:
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus = Queue()
        for i in range(num_gpus):
            self.available_gpus.put(i)
        self.user_sessions: Dict[str, UserSession] = {}
        self.lock = threading.Lock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def acquire_gpu(self, user_id: str) -> int:
        with self.lock:
            # Check if user already has a session
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                session.last_activity = datetime.now()
                return session.gpu_id
            
            # Try to get a new GPU
            if not self.available_gpus.empty():
                gpu_id = self.available_gpus.get()
                self.user_sessions[user_id] = UserSession(
                    user_id=user_id,
                    gpu_id=gpu_id,
                    last_activity=datetime.now()
                )
                logging.info(f"Assigned GPU {gpu_id} to user {user_id}")
                return gpu_id
            
            return None
    
    def release_gpu(self, user_id: str) -> None:
        with self.lock:
            if user_id in self.user_sessions:
                session = self.user_sessions.pop(user_id)
                self.available_gpus.put(session.gpu_id)
                logging.info(f"Released GPU {session.gpu_id} from user {user_id}")
    
    def _cleanup_sessions(self) -> None:
        while True:
            time.sleep(60)  # Check every minute
            with self.lock:
                current_time = datetime.now()
                expired_users = [
                    user_id for user_id, session in self.user_sessions.items()
                    if (current_time - session.last_activity).total_seconds() > SESSION_TIMEOUT
                ]
                for user_id in expired_users:
                    self.release_gpu(user_id)
                    logging.info(f"Cleaned up expired session for user {user_id}")

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Dict[torch.device, BarkModel]] = {}
        self.processors: Dict[str, AutoProcessor] = {}
        self.load_models()

    def load_models(self):
        try:
            for model_name in ["suno/bark", "suno/bark-small"]:
                self.models[model_name] = {}
                self.processors[model_name] = AutoProcessor.from_pretrained(model_name)
                for device in devices:
                    model = BarkModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16
                    ).to(device)
                    model.eval()
                    self.models[model_name][device] = model
            logging.info("Models loaded successfully on all devices")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def get_model(self, model_name: str, device: torch.device) -> BarkModel:
        return self.models[model_name][device]

    def get_processor(self, model_name: str) -> AutoProcessor:
        return self.processors[model_name]

model_manager = ModelManager()
gpu_manager = GPUManager(num_gpus)

def generate_speech(text: str, model_name: str, voice_preset: str, device: torch.device) -> np.ndarray:
    try:
        model = model_manager.get_model(model_name, device)
        processor = model_manager.get_processor(model_name)
        
        inputs = processor(text, voice_preset=voice_preset)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = model.generate(**inputs)
            
        audio_array = audio_array.cpu().numpy().squeeze()
        if audio_array.size == 0:
            logging.warning(f"Generated audio is empty for text: {text}")
            return np.array([])
            
        return np.clip(audio_array, -1.0, 1.0)
    except Exception as e:
        logging.error(f"Error in speech generation: {e}")
        raise

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_voice():
    try:
        user_id = session.get('user_id')
        text = request.form.get("text")
        model_name = request.form.get("model_name", "suno/bark")
        voice_preset = request.form.get("voice_preset", "v2/en_speaker_0")

        if not text:
            raise ValueError("Missing required parameter: text")

        # Try to acquire a GPU
        gpu_id = gpu_manager.acquire_gpu(user_id)
        if gpu_id is None:
            return jsonify({
                "success": False,
                "error": "All GPUs are currently in use. Please try again later."
            }), 503

        try:
            device = devices[gpu_id]
            timestamp = int(time.time())
            output_filename = f"generated_voice_{timestamp}.wav"
            output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

            # Generate speech
            audio_data = generate_speech(text, model_name, voice_preset, device)
            
            # Convert to int16 format
            audio_data = (audio_data * 32767).astype(np.int16)

            # Save the audio file
            write(output_path, SAMPLE_RATE, audio_data)
            logging.info(f"Audio file saved at {output_path}")

            return jsonify({
                "success": True,
                "file_path": f"/static/audio/{output_filename}",
                "message": "Audio generated successfully"
            })
        finally:
            # Don't release the GPU here - it stays allocated to the user
            pass

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error generating voice: {error_msg}")
        if 'user_id' in locals():
            gpu_manager.release_gpu(user_id)  # Release GPU on error
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

@app.route("/release", methods=["POST"])
def release_session():
    """Endpoint for clients to explicitly release their GPU when done"""
    user_id = session.get('user_id')
    if user_id:
        gpu_manager.release_gpu(user_id)
        return jsonify({"success": True, "message": "Session released"})
    return jsonify({"success": False, "error": "No active session"}), 400

@app.route("/status", methods=["GET"])
def get_status():
    """Get current GPU allocation status"""
    with gpu_manager.lock:
        available = gpu_manager.available_gpus.qsize()
        in_use = num_gpus - available
        return jsonify({
            "total_gpus": num_gpus,
            "available_gpus": available,
            "gpus_in_use": in_use
        })

if __name__ == "__main__":
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    logging.info(f"Starting Flask application with {num_gpus} GPUs")
    app.run(host="0.0.0.0", port=5000, debug=False)  # Debug mode off for thread safety
