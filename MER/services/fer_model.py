
import tensorflow as tf


model = tf.keras.models.load_model(r"MER\services\emotion_model_final.h5")


def detect_emotions(face):
    return model.predict(face)


