import os
import json
import numpy as np
import pandas as pd

from features import extract_features

# Load metadata
with open("dataset/audiomnist/audioMNIST_meta.txt", "r") as f:
    metadata = json.load(f)

data = []

audio_root = "dataset/audiomnist/data"

count = 0

for speaker_id in metadata:

    speaker_folder = os.path.join(audio_root, speaker_id)

    if not os.path.exists(speaker_folder):
        continue

    age = int(metadata[speaker_id]["age"])
    gender = metadata[speaker_id]["gender"]

    for file in os.listdir(speaker_folder):

        if file.endswith(".wav"):

            audio_path = os.path.join(speaker_folder, file)

            try:
                features = extract_features(audio_path)

                row = list(features)

                row.append(age)
                row.append(gender)

                data.append(row)

                count += 1

                if count % 1000 == 0:
                    print("Processed:", count)

            except Exception as e:
                print("Error:", audio_path)

df = pd.DataFrame(data)

df.to_csv("audiomnist_features.csv", index=False)

print("\nSaved!")
print("Samples:", len(df))