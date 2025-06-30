import torch
import numpy as np
from models.lstm_model import SentimentLSTM

# ======= Hyperparameters ========
MAX_LEN = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Load vocab & model ========
vocab = np.load('models/vocab.npy', allow_pickle=True).item()

model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    output_dim=1,
    n_layers=2,
    bidirectional=True,
    dropout=0.3,
    pad_idx=vocab['<pad>']
)
model.load_state_dict(torch.load('models/lstm_model.pth', map_location=device))
model = model.to(device)
model.eval()

def encode(text):
    return [vocab.get(word, vocab['<unk>']) for word in text.lower().split()]

def predict_sentiment(text):
    encoded = encode(text)[:MAX_LEN]
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        label = "positive" if prob >= 0.5 else "negative"
    return label, prob

if __name__ == "__main__":
    text = input("Enter a review: ")
    label, prob = predict_sentiment(text)
    print(f"Sentiment: {label} ({prob:.2f})")
