from features import extract_features

audio_file = r"dataset/ravdess/Actor_01/03-01-01-01-01-01-01.wav"

features = extract_features(audio_file)

print("Feature Shape:", features.shape)
print(features)