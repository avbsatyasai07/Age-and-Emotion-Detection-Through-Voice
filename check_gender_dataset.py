import pandas as pd

df = pd.read_csv("dataset/voice.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nLabels:")
print(df["label"].value_counts())