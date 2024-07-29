
import cv2
import time
from MER.services import fer_model as emotion_detector  
import numpy as np
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    prediction = emotion_detector.detect_emotions(face=face)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion



def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect emotions in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face)
                # Predict emotion

                color = (0, 0, 255) if emotion == 'Angry' else \
                        (0, 255, 0) if emotion == 'Happy' else \
                        (255, 0, 0) if emotion == 'Sad' else \
                        (255, 255, 0) if emotion == 'Surprise' else \
                        (0, 255, 255) if emotion == 'Neutral' else \
                        (255, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                   
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)