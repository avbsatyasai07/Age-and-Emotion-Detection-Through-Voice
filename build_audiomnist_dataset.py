import os
import json

# Load metadata
with open("dataset/audiomnist/audioMNIST_meta.txt", "r") as f:
    metadata = json.load(f)

print("Total Speakers:", len(metadata))

# Example speaker
print("\nSpeaker 01:")
print(metadata["01"])

# Count WAV files
total_wavs = 0

data_path = "dataset/audiomnist/data"

for speaker in os.listdir(data_path):

    speaker_path = os.path.join(data_path, speaker)

    if os.path.isdir(speaker_path):

        wavs = [f for f in os.listdir(speaker_path)
                if f.endswith(".wav")]

        total_wavs += len(wavs)

print("\nTotal WAV files:", total_wavs)