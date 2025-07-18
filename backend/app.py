"""
This module implements a Flask API for fake profile detection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_service import ModelService

app = Flask(__name__)
CORS(app)
model_service = ModelService()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts if a Twitter profile is fake or real based on the provided data.
    """
    screen_name = request.form.get("screen_name")
    description = request.form.get("description", "")
    followers_count = int(request.form.get("followers_count", 0))
    friends_count = int(request.form.get("friends_count", 0))
    statuses_count = int(request.form.get("statuses_count", 0))

    if not screen_name:
        return jsonify({"error": "Screen name is required"}), 400

    data = {
        "description": description,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "statuses_count": statuses_count
    }

    result = model_service.predict(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
