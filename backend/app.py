# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_service import ModelService

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
model_service = ModelService()

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    screen_name = request.form.get("screen_name")
    description = request.form.get("description", "")
    profile_pic = request.files.get("profile-pic")
    followers_count = int(request.form.get("followers_count", 0))
    friends_count = int(request.form.get("friends_count", 0))
    statuses_count = int(request.form.get("statuses_count", 0))
    verified = request.form.get("verified", "False")

    # Validate
    if not screen_name:
        return jsonify({"error": "Screen name is required"}), 400

    # Prepare data
    data = {
        "profile_pic": "Yes" if profile_pic else "No",
        "followers_count": followers_count,
        "friends_count": friends_count,
        "statuses_count": statuses_count,
        "verified": verified
    }

    # Get prediction
    result = model_service.predict(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)