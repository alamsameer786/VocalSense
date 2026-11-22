from pathlib import Path
import tempfile
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from .config import MODEL_PATH, MAX_UPLOAD_MB, ALLOWED_EXTENSIONS
from .model import EmotionModel

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
model = EmotionModel(MODEL_PATH)

@app.get('/')
def index():
    return render_template('index.html', model_available=model.available)

@app.get('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model.available, 'model_path': str(MODEL_PATH)})

@app.post('/api/predict')
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file supplied. Use the audio field.'}), 400
    audio = request.files['audio']
    if not audio.filename:
        return jsonify({'error': 'Please select or record a WAV file.'}), 400
    suffix = Path(secure_filename(audio.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Only WAV audio is accepted by this version.'}), 415
    if not model.available:
        return jsonify({'error': 'Emotion model is not trained yet. Run python train_model.py with the RAVDESS dataset.'}), 503

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f'input{suffix}'
        audio.save(path)
        try:
            label, confidence, probabilities = model.predict(str(path))
        except Exception as exc:
            return jsonify({'error': f'Could not analyze audio: {exc}'}), 422
    ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
        'emotion': label,
        'confidence': round(confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in ranked},
    })

@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'Audio is too large. Maximum is {MAX_UPLOAD_MB} MB.'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
