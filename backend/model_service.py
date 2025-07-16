# backend/model_service.py
import joblib
import pandas as pd

class ModelService:
    def __init__(self):
        self.model = joblib.load("twitter_model.pkl")
        self.le_profile_pic = joblib.load("le_profile_pic.pkl")
        self.le_verified = joblib.load("le_verified.pkl")

    def preprocess(self, data):
        df = pd.DataFrame([data])
        df["profile_pic"] = self.le_profile_pic.transform([data["profile_pic"]])[0]
        df["verified"] = self.le_verified.transform([data["verified"]])[0]
        return df[["profile_pic", "followers_count", "friends_count", "statuses_count", "verified"]]

    def predict(self, data):
        X = self.preprocess(data)
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0][prediction] * 100
        return {"prediction": "real" if prediction == 0 else "fake", "confidence": confidence}