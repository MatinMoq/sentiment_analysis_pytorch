import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.lstm_model import SentimentLSTM
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
import os

# ======= Hyperparameters ========
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
N_EPOCHS = 5
MAX_LEN = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Data Processing ========
print("Loading dataset...")
dataset = load_dataset("imdb")

# Build vocab
all_texts = [x['text'] for x in dataset['train']]
counter = Counter()
for line in all_texts:
    counter.update(line.lower().split())
vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(20000))}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

def encode(text):
    return [vocab.get(word, vocab['<unk>']) for word in text.lower().split()]

def collate_batch(batch):
    texts = [torch.tensor(encode(x['text'])[:MAX_LEN]) for x in batch]
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.float32)
    padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return padded, labels

train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ======= Model, Optimizer, Loss ========
model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT,
    pad_idx=vocab['<pad>']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)

# ======= Training Loop ========
def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    return (rounded.squeeze() == y).float().mean()

print("Training...")
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for text, labels in train_loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={epoch_acc/len(train_loader):.4f}')
    # Optionally: Add validation loop here

# Save model & vocab
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/lstm_model.pth')
np.save('models/vocab.npy', vocab)
print("Training completed and model saved.")
