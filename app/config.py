import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv('MODEL_PATH', BASE_DIR / 'models' / 'emotion_model.joblib'))
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '25'))
ALLOWED_EXTENSIONS = {'.wav'}
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
