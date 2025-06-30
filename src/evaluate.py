import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models.lstm_model import SentimentLSTM

# ======= Hyperparameters ========
BATCH_SIZE = 64
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

# ======= Data Preparation ========
def encode(text):
    return [vocab.get(word, vocab['<unk>']) for word in text.lower().split()]

def collate_batch(batch):
    texts = [torch.tensor(encode(x['text'])[:MAX_LEN]) for x in batch]
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.float32)
    padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return padded, labels

dataset = load_dataset("imdb")
test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ======= Evaluation ========
def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    return (rounded.squeeze() == y).float().mean()

total_loss = 0
total_acc = 0
criterion = torch.nn.BCEWithLogitsLoss().to(device)

with torch.no_grad():
    for text, labels in test_loader:
        text, labels = text.to(device), labels.to(device)
        preds = mode
