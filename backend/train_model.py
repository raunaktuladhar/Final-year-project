# backend/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load datasets
fusers = pd.read_csv("fusers.csv")
users = pd.read_csv("users.csv")

# Add labels
fusers["label"] = 1  # Fake
users["label"] = 0   # Real
data = pd.concat([fusers, users], ignore_index=True)

# Preprocess features
# Create profile_pic from default_profile_image and profile_image_url
data["profile_pic"] = data.apply(lambda x: "No" if x["default_profile_image"] == 1 else "Yes" if x["profile_image_url"] else "No", axis=1)

# Encode categorical features
le_profile_pic = LabelEncoder()
data["profile_pic"] = le_profile_pic.fit_transform(data["profile_pic"])
le_verified = LabelEncoder()
data["verified"] = le_verified.fit_transform(data["verified"].fillna("False"))

# Save encoders for model_service.py
joblib.dump(le_profile_pic, "le_profile_pic.pkl")
joblib.dump(le_verified, "le_verified.pkl")

# Select features
X = data[["profile_pic", "followers_count", "friends_count", "statuses_count", "verified"]]
y = data["label"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "twitter_model.pkl")