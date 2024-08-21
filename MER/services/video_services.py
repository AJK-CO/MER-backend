
import cv2
import time
from MER.services import fer_model as emotion_detector
import numpy as np
import os
import json

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

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

                color = {
                    'Angry': (0, 0, 255),
                    'Disgust': (153, 50, 204),  # Dark orchid
                    'Fear': (255, 165, 0),  # Orange
                    'Happy': (0, 255, 0),
                    'Sad': (255, 0, 0),
                    'Surprise': (255, 255, 0),
                    'Neutral': (0, 255, 255),
                }.get(emotion, (255, 0, 255))  # Default to magenta

                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                   
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def get_fer_emotion():
    video_path = os.path.join(os.getcwd(), "videos", "recording.mp4")
    camera = cv2.VideoCapture(video_path)
    emotions = []
    
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            timestamp = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Timestamp in seconds
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face)
                
                # Store emotion with timestamp
                emotions.append({
                    'time': timestamp,
                    'emotion': emotion
                })

            time.sleep(0.03)

    # Calculate average emotion for the video
    if emotions:
        emotion_counts = {emotion: 0 for emotion in ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']}
        for data in emotions:
            emotion_counts[data['emotion']] += 1

        avg_emotion = max(emotion_counts, key=emotion_counts.get)
    else:
        avg_emotion = 'Neutral'

    # Return data as JSON
    result = {
        'total_frames': total_frames,
        'frame_emotions': emotions,
        'average_emotion': avg_emotion
    }

    return json.dumps(result) 

if __name__=="__main__":
# Example usage:
    response = get_fer_emotion()
    print(response)

