from emotion_detection import predict_emotion

audio_file = r"dataset/ravdess/Actor_01/03-01-03-01-01-01-01.wav"

emotion = predict_emotion(audio_file)

print("Predicted Emotion:", emotion)