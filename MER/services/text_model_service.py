import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from nltk.corpus import stopwords
import nltk
import pickle

# Load the saved model
model = load_model('MER/services/text_model.keras')
with open('MER/services/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess the input text in the same way as during training
def preprocess_text(text):
    
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # Step 1: Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Step 2: Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Remove numeric values
    text = re.sub(r'\d+', '', text)

    # Step 5: Lowercasing
    text = text.lower()

    # Step 6: Remove stop words
    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop])

    # Step 7: Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

def predict_text_emotion(input_text):
    # Preprocess new texts

    # Tokenize the new texts (Note: Use the same tokenizer as used during training)
    
   
    # processed_text = preprocess_text(input_text)
    
    # Tokenize the text
    preprocessed_text = preprocess_text(input_text)
    
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([preprocessed_text])
    
    # Pad the sequence
    maxlen = 79  # Use the same maxlen used during training
    padded_sequence = pad_sequences(sequences, maxlen=maxlen, padding='post')
    
    # Predict
    prediction = model.predict(padded_sequence)

    # Assuming a binary classification (use argmax for multi-class)
    predicted_labels =np.argmax(prediction, axis=-1)[0]


    print(f"Predicted Label: {predicted_labels}\n")
    label_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
    }

    return label_mapping[predicted_labels]
