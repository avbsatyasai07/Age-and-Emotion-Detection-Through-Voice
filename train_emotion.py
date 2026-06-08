import os
import numpy as np
from features import extract_features

X = []
y = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

dataset_path = "dataset/ravdess"

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]

            emotion = emotion_map[emotion_code]

            audio_path = os.path.join(actor_path, file)

            features = extract_features(audio_path)

            X.append(features)
            y.append(emotion)

print("Samples:", len(X))
print("Labels:", len(y))
print("Emotions:", set(y))