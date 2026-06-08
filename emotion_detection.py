import pickle
from features import extract_features

with open("models/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_emotion(audio_path):

    features = extract_features(audio_path)

    prediction = model.predict([features])[0]

    return prediction