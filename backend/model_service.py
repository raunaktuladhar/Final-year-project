"""
This module provides a service for using the fake profile detection model.
"""
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np


class BertLSTM(nn.Module):
    """A simple LSTM model to be used with BERT embeddings."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """Initializes the BertLSTM model."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """The forward pass of the model."""
        x = x.unsqueeze(1)
        h0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ModelService:
    """A service to load and use the trained model for predictions."""

    def __init__(self):
        """Initializes the service and loads the models."""
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert_model")
        self.bert_model.eval()

        # Correct input_size is BERT embedding size + number of numerical features
        self.lstm_model = BertLSTM(
            input_size=768 + 3, hidden_size=64, num_layers=2, num_classes=2
        )
        self.lstm_model.load_state_dict(torch.load("bert_lstm_model.pt"))
        self.lstm_model.eval()

    def get_bert_embedding(self, text):
        """Generates BERT embeddings for the given text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def predict(self, data):
        """Predicts if a profile is fake or real."""
        bert_embedding = self.get_bert_embedding(data["description"])
        numerical_features = np.array([
            data["followers_count"],
            data["friends_count"],
            data["statuses_count"],
        ])

        combined_features = np.hstack((bert_embedding, numerical_features))
        features_tensor = torch.tensor(
            combined_features, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            out = self.lstm_model(features_tensor)
            probs = torch.softmax(out, dim=1)
            prediction = probs.argmax(dim=1).item()
            confidence = probs[0][prediction].item() * 100

        return {
            "prediction": "real" if prediction == 0 else "fake",
            "confidence": confidence,
        }
