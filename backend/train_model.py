"""
This script trains a fake profile detection model.

It uses BERT embeddings for text features and an LSTM for classification.
"""
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel


# Load datasets
try:
    fusers = pd.read_csv("fusers.csv")
    users = pd.read_csv("users.csv")
except FileNotFoundError as e:
    print(
        f"Error: {e}. Please ensure fusers.csv and users.csv are in the backend directory."
    )
    sys.exit(1)

# Add labels
fusers["label"] = 1  # Fake
users["label"] = 0   # Real
data = pd.concat([fusers, users], ignore_index=True).fillna(0)

# BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False

def get_bert_embeddings(descriptions):
    """Extracts BERT embeddings for a list of descriptions."""
    embeddings = []
    with torch.no_grad():
        for desc in descriptions:
            desc = str(desc) if desc is not None else ""
            inputs = tokenizer(
                desc, return_tensors="pt", truncation=True, padding=True, max_length=128
            )
            outputs = bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

bert_embeddings = get_bert_embeddings(data["description"])

# Combine features
numerical_features = data[
    ["followers_count", "friends_count", "statuses_count"]
].values
features = np.hstack((bert_embeddings, numerical_features))
labels = data["label"].values


class CombinedDataset(Dataset):
    """A custom dataset to handle combined text and numerical features."""

    def __init__(self, input_features, input_labels):
        """Initializes the dataset."""
        self.features = torch.tensor(input_features, dtype=torch.float32)
        self.labels = torch.tensor(input_labels, dtype=torch.long)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset."""
        return self.features[idx], self.labels[idx]


dataset = CombinedDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class BertLSTM(nn.Module):
    """A simple LSTM model to be used with BERT embeddings."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """Initializes the BertLSTM model."""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """The forward pass of the model."""
        # Reshape for LSTM: (batch, seq_len, input_size)
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = BertLSTM(
    input_size=features.shape[1], hidden_size=64, num_layers=2, num_classes=2
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train LSTM
model.train()
for epoch in range(10):
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        model_outputs = model(batch_features)
        loss = criterion(model_outputs, batch_labels)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "bert_lstm_model.pt")
bert_model.save_pretrained("bert_model")  # Save fine-tuned BERT
tokenizer.save_pretrained("bert_model")

print("BERT and LSTM models trained and saved.")
