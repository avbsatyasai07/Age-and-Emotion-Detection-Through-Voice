import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("audiomnist_features.csv")

X = df.iloc[:, :-2]
y = df.iloc[:, -2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("Age MAE:", mae)

with open("models/age_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Age Model Saved!")