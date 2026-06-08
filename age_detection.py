import pickle
from features import extract_features

with open("models/age_model.pkl", "rb") as f:
    age_model = pickle.load(f)

def predict_age(audio_path):

    features = extract_features(audio_path)

    age = age_model.predict([features])[0]

    return round(age)