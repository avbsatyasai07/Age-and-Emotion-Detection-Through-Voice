import pickle
from features import extract_features

audio_file = r"dataset/audiomnist/data/01/0_01_0.wav"

features = extract_features(audio_file)

# Gender
with open("models/gender_audio_model.pkl", "rb") as f:
    gender_model = pickle.load(f)

gender = gender_model.predict([features])[0]

# Age
with open("models/age_model.pkl", "rb") as f:
    age_model = pickle.load(f)

age = age_model.predict([features])[0]

print("Gender:", gender)
print("Age:", round(age))