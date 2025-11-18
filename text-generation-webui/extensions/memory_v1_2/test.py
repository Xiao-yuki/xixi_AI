import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(BASE_DIR, 'words.json')

with open(TEST_FILE, "r", encoding="utf-8") as f:
    token2idx = json.load(f)
#tokens = ["明", "天", "吃", "壽", "司"]
#indices = [token2idx[t] for t in tokens]

class Embedding:
    def __init__(self, vocab_size, embed_dim, lr=0.001):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
        self.lr = lr
        self.last_indices = None
    
    def forward(self, indices):
        self.last_indices = indices
        return self.W[indices]
    
    def backward(self,grad_output):
        for i, idx in enumerate(self.last_indices):
            self.W[idx] = self.W[idx] - self.lr * grad_output[i]
    
class BiRNN:
    def __init__(self, input_dim, hidden_dim, lr=0.001):
        self.hidden_dim = hidden_dim
        self.lr = lr
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
            h_b[t] = h_prev

        return np.concatenate([h_f, h_b], axis=1)
    
    def backward(self, x_seq, grad_output):
        T = len(x_seq)
        grad_f = grad_output[:, :self.hidden_dim]
        grad_b = grad_output[:, self.hidden_dim:]
        grad_input = np.zeros_like(x_seq)
        #forward
        for t in range(T):
            x = x_seq[t:t+1]
            dh = grad_f[t:t+1]
            self.Wx_f = self.Wx_f - self.lr * (x.T @ dh)
            self.b_f = self.b_f - self.lr * dh
            grad_input[t:t+1] = grad_input[t:t+1] + dh @ self.Wx_f.T
        #backward
        for t in range(T):
            x = x_seq[t:t+1]
            dh = grad_b[t:t+1]
            self.Wx_b = self.Wx_b - self.lr * (x.T @ dh)
            self.b_b = self.b_b - self.lr * dh
            grad_input[t:t+1] = grad_input[t:t+1] + dh @ self.Wx_b.T

        return grad_input
    
class Classifier:
    def __init__(self, input_dim, lr=0.001):
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

def tolenize_and_label(sentence, keywords, token2idx):
    tokens = list(sentence)
    indices = [token2idx[t] for t in tokens]
    labels = np.array([[1 if t in keywords else 0] for t in tokens])
    return tokens, indices, labels

#test
embed = Embedding(vocab_size=5000, embed_dim=16)
encoder = BiRNN(input_dim=16, hidden_dim=32, lr=0.001)
clf = Classifier(input_dim=64)

#TRAINING_DATA = os.path.join(BASE_DIR, 'training_data.json')
#with open(TRAINING_DATA, "r", encoding="utf-8") as d:
#    training_data = json.load(d)

training_data = [
    ("我明天想吃壽司", ["吃", "壽司"]),
    ("明天要去打籃球", ["籃球"]),
    ("貓咪在沙發上睡覺", ["貓咪", "睡覺"]),
]

for sentence, keywords in training_data:
    tokens, indices, labels = tolenize_and_label(sentence, keywords, token2idx)

    for epoch in range(30000):
        x_embed = embed.forward(indices)
        h_seq = encoder.forward(x_embed)
        preds = clf.forward(h_seq)

        loss = binary_cross_entropy(preds, labels)

        grad_trom_clf = clf.backward(labels)
        grad_to_rnn = encoder.backward(x_embed, grad_trom_clf)
        embed.backward(grad_to_rnn)

        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, preds = {preds.ravel()}")

        if loss < 0.3:
            break

    print("預測分數：", preds.ravel())
    print("Loss：", loss)
    print("關鍵字：", extract_keywords(tokens, preds.ravel()))