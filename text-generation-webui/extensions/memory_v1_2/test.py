import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(BASE_DIR, 'words.json')

with open(TEST_FILE, "r", encoding="utf-8") as f:
    token2idx = json.load(f)
tokens = ["明", "天", "吃", "壽", "司"]
indices = [token2idx[t] for t in tokens]

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def forward(self, indices):
        return self.W[indices]
    
class BiRNN:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        #forward
        self.Wx_f = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Wh_f = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros((1, hidden_dim))
        #backward
        self.Wx_b = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Wh_b = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_b = np.zeros((1, hidden_dim))

    def step(self, x, h_prev, Wx, Wh, b):
        return np.tanh(x @ Wx + h_prev @ Wh + b)
    
    def forward(self, x_seq):
        T = x_seq.shape[0]
        h_f = np.zeros((T, self.hidden_dim))
        h_b = np.zeros((T, self.hidden_dim))
        h_prev = np.zeros((1, self.hidden_dim))
        #forward
        for t in range(T):
            h_prev = self.step(x_seq[t:t+1], h_prev, self.Wx_f, self.Wh_f, self.b_f)
            h_f[t] = h_prev
        #backward
        h_prev = np.zeros((1, self.hidden_dim))
        for t in reversed(range(T)):
            h_prev = self.step(x_seq[t:t+1], h_prev, self.Wx_b, self.Wh_b, self.b_b)

        return np.concatenate([h_f, h_b], axis=1)
    
class Classifier:
    def __init__(self, input_dim, lr=0.1):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0
        self.lr = lr

    def forward(self, h_seq):
        self.h_seq = h_seq
        z = h_seq @ self.W + self.b
        self.preds = 1 / (1 + np.exp(-z))
        return self.preds
    
    def backward(self, labels):
        dz = self.preds - labels
        dw = self.h_seq.T @ dz
        db = np.sum(dz)

        dh = dz @ self.W.T

        self.W -= self.lr * dw
        self.b -= self.lr * db

        return dh
    
def binary_cross_entropy(preds, targets):
    eps = 1e-7
    preds = np.clip(preds, eps, 1 - eps)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

def extract_keywords(tokens , scores, threshold=0.5):
    return [tok for tok, s in zip(tokens, scores) if s > threshold]

#test
embed = Embedding(vocab_size=5000, embed_dim=16)
encoder = BiRNN(input_dim=16, hidden_dim=32)
clf = Classifier(input_dim=64)

tokens = ["吃", "壽", "司"]
indices = [token2idx[t] for t in tokens]
labels = np.array([[1], [1], [1]])

for epoch in range(10):
    x_embed = embed.forward(indices)
    h_seq = encoder.forward(x_embed)
    preds = clf.forward(h_seq)

    loss = binary_cross_entropy(preds, labels)

    grad_trom_clf = clf.backward(labels)

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, preds = {preds.ravel()}")

print("預測分數：", preds.ravel())
print("Loss：", loss)
print("關鍵字：", extract_keywords(tokens, preds.ravel()))