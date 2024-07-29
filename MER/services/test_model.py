import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from pydub import AudioSegment

def verify_audio(file_path):
    try:
        audio = AudioSegment.from_wav(file_path)
        print("Audio file loaded successfully.")
        return True
    except Exception as e:
        print("Error loading audio file:", e)
        return False

def extract_features(file_path):
    if not verify_audio(file_path):
        print("Audio verification failed.")
        return None

    try:
        # Load audio file
        audio, sr = sf.read(file_path)
        print(f"Audio data: {audio[:5]}...")  # Print first few samples for debug
        print(f"Sample rate: {sr}")
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        return np.hstack((mfccs, chroma, mel, contrast, tonnetz))
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

def predict_emotion(file_path, model, labels):
    """Predict the emotion of a given speech file."""
    # Extract features from the audio file
    features = extract_features(file_path)
    if features is None:
        print("Could not extract features from the audio file.")
        return None

    # Reshape the features to match the model input
    features = features.reshape(1, -1)
    # Make a prediction
    prediction = model.predict([features, features])
    # Get the label with the highest probability
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

def find_emotion():
    # Load the trained model using tf.keras
    model = tf.keras.models.load_model(r"MER\services\speech.keras")

    # Define the emotion labels
    labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    # Path to the audio file to be tested (adjust the path according to your file location)
    audio_file = r"MER\services\recording.wav"
    # audio_file="MER/services/OAF_back_fear.wav"
    # Predict the emotion
    predicted_emotion = predict_emotion(audio_file, model, labels)

    if predicted_emotion:
        print(predict_emotion)
        return predicted_emotion
    else:
        return "Failed to predict the emotion."

if __name__ == "__main__":
    emotion = find_emotion()
    print("Predicted Emotion:", emotion)
