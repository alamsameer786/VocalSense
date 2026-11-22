from app.main import app

def test_home():
    c=app.test_client(); r=c.get('/'); assert r.status_code==200; assert b'VocalSense' in r.data

def test_health():
    c=app.test_client(); r=c.get('/api/health'); assert r.status_code==200; assert r.json['status']=='ok'

def test_predict_requires_audio():
    c=app.test_client(); r=c.post('/api/predict'); assert r.status_code==400
