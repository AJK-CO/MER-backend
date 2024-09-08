import numpy as np
from tensorflow.keras.models import load_model
from test_model import find_emotion
from video_services import get_fer_emotion
from speech_to_text_whisper import get_text
from text_model_service import predict_text_emotion
import json

# Define your emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def emotion_to_vector(emotion, labels):
    """Convert emotion string to one-hot encoded vector."""
    vector = np.zeros(len(labels))
    if emotion in labels:
        vector[labels.index(emotion)] = 1
    return vector

# Preprocess the input data
facial_emotion = json.loads(get_fer_emotion())
text = get_text("Audios/recording.wav") 

# Speech model prediction
speech_prediction = find_emotion("Audios/recording.wav")

# Text model prediction
text_prediction = predict_text_emotion(text)

# Convert predictions to one-hot encoded vectors
facial_prediction_vector = emotion_to_vector(facial_emotion, emotion_labels)
speech_prediction_vector = emotion_to_vector(speech_prediction, emotion_labels)
text_prediction_vector = emotion_to_vector(text_prediction, emotion_labels)

# Print shapes and types of predictions for debugging
print(f"Facial Prediction Vector: {facial_prediction_vector}")
print(f"Speech Prediction Vector: {speech_prediction_vector}")
print(f"Text Prediction Vector: {text_prediction_vector}")

# Ensure predictions are properly shaped for input to the model
facial_prediction = np.expand_dims(facial_prediction_vector, axis=0)  # Add batch dimension
speech_prediction = np.expand_dims(speech_prediction_vector, axis=0)  # Add batch dimension
text_prediction = np.expand_dims(text_prediction_vector, axis=0)  # Add batch dimension

# Print shapes of each prediction to ensure they are valid inputs
print(f"Facial Prediction Shape: {facial_prediction.shape}")
print(f"Speech Prediction Shape: {speech_prediction.shape}")
print(f"Text Prediction Shape: {text_prediction.shape}")

# Load the fusion model and make the final prediction
fusion_model = load_model(r'C:\Users\16307\Desktop\term\MER-backend\MER\services\multimodal_fusion_model.h5')

# Pass the inputs separately to the fusion model
final_prediction = fusion_model.predict([facial_prediction, speech_prediction, text_prediction])

# Get the predicted emotion
predicted_emotion = emotion_labels[np.argmax(final_prediction, axis=1)[0]]

# Print the outputs
print(f"Facial Prediction: {emotion_labels[np.argmax(facial_prediction)]}")
print(f"Speech Prediction: {emotion_labels[np.argmax(speech_prediction)]}")
print(f"Text Prediction: {text_prediction}")
print(f"Final MER Prediction: {predicted_emotion}")
