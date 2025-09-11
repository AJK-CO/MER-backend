import numpy as np
from tensorflow.keras.models import load_model

# Define emotion labels used by all models
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def emotion_to_vector(emotion, labels):
    """Convert emotion string to one-hot encoded vector."""
    vector = np.zeros(len(labels))
    if emotion in labels:
        vector[labels.index(emotion)] = 1
    return vector

# Step 1: Get predictions from individual models (these should return strings)

def predict_mer(facial_emotion,speech_emotion,text_emotion):
    # Step 2: Convert string predictions to one-hot encoded vectors
    facial_vector = emotion_to_vector(facial_emotion, emotion_labels)
    speech_vector = emotion_to_vector(speech_emotion, emotion_labels)
    text_vector = emotion_to_vector(text_emotion, emotion_labels)

    # Step 3: Ensure all predictions are shaped correctly for input to the fusion model
    facial_vector = np.expand_dims(facial_vector, axis=0)  # Add batch dimension
    speech_vector = np.expand_dims(speech_vector, axis=0)  # Add batch dimension
    text_vector = np.expand_dims(text_vector, axis=0)      # Add batch dimension

    # Step 4: Concatenate the predictions (late fusion) or pass them separately based on your fusion model
    fused_input = np.concatenate([facial_vector, speech_vector, text_vector], axis=1)

    # Step 5: Load your fusion model
    fusion_model = load_model(r'MER\services\multimodal_fusion_model_new_1.keras')

    # Step 6: Predict the final emotion based on fused input
    final_prediction = fusion_model.predict(fused_input)

    # Step 7: Get the predicted emotion label
    predicted_emotion = emotion_labels[np.argmax(final_prediction, axis=1)[0]]

    # Step 8: Print the results
    return predicted_emotion