from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("road_accident_dataset.csv")
MODEL_PATH = Path("accident_severity_model.joblib")

if not DATA_PATH.exists():
    raise FileNotFoundError("road_accident_dataset.csv not found. Keep it in the project folder.")

df = pd.read_csv(DATA_PATH)
target = "accident_severity"
X = df.drop(columns=[target])
y = df[target]

categorical = ["road_type", "weather", "light", "traffic_density", "road_surface", "alcohol_involvement"]
numeric = ["speed_limit", "vehicles_involved", "driver_age"]

preprocess = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Model training completed.")
print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(pipeline, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")
