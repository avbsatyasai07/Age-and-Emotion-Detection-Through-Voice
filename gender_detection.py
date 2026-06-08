import pickle
from features import extract_features

with open("models/gender_audio_model.pkl", "rb") as f:
    gender_model = pickle.load(f)

def predict_gender(audio_path):

    features = extract_features(audio_path)

    gender = gender_model.predict([features])[0]

    return gender