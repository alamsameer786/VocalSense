from pathlib import Path
import joblib
import numpy as np
from .audio_features import extract_features

class EmotionModel:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metadata = {}
        self.load()

    def load(self):
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.pipeline = bundle['model'] if isinstance(bundle, dict) else bundle
            self.metadata = bundle.get('metadata', {}) if isinstance(bundle, dict) else {}

    @property
    def available(self):
        return self.pipeline is not None

    def predict(self, audio_path: str):
        if not self.available:
            raise RuntimeError('Model is not trained. Run: python train_model.py')
        x = extract_features(audio_path).reshape(1, -1)
        label = str(self.pipeline.predict(x)[0])
        probabilities = {}
        if hasattr(self.pipeline, 'predict_proba'):
            probs = self.pipeline.predict_proba(x)[0]
            for cls, p in zip(self.pipeline.classes_, probs):
                probabilities[str(cls)] = float(p)
        confidence = probabilities.get(label, 0.0)
        return label, confidence, probabilities
