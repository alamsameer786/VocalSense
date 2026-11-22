Steps to run the project.
a. Clone the project from github in your system
b. go in SpeechEmotionRecognizer by using cd cmd
c. then run cd .\setup_windows.bat
d. install all the dependencies pip install -r requirements.txt
e. if you want to train the model again use this command python train_model.py --dataset data/ravdess --output models/emotion_model.joblib otherwise already have trained model in the file.
f. to run the server use command python -m app and go in http://127.0.0.1:5000 you can see the project running in local server