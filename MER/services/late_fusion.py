import tensorflow as tf
from tensorflow.keras.models import load_model




def get_mer_output(speech_output,facial_output,text_output):


    # Concatenate the outputs
    fused_output = tf.concat([speech_output, facial_output, text_output], axis=1)

    # Define a fusion layer
    fusion_layer = tf.keras.layers.Dense(7, activation='softmax')  # num_classes is the number of emotion categories
    final_output = fusion_layer(fused_output)

    # Apply softmax to get final predictions (softmax already applied in fusion layer)
    final_prediction = final_output

    # Interpret the prediction
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  # Replace with your emotion labels
    predicted_emotion = emotion_labels[tf.argmax(final_prediction, axis=1).numpy()[0]]

    # Use the predicted emotion in customer service logic
    return predicted_emotion
