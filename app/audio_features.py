import numpy as np
import librosa

SR = 22050
N_MFCC = 40


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        raise ValueError('Audio file is empty.')
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size < 2048:
        y = np.pad(y, (0, 2048 - y.size))

    feats = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend(np.mean(mel_db, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    for arr in (zcr, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff):
        feats.extend([float(np.mean(arr)), float(np.std(arr))])

    result = np.asarray(feats, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError('Audio produced invalid numerical features.')
    return result
