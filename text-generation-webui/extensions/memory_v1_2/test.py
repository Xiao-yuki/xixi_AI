import numpy as np

token2idx = {"明": 0, "天": 1, "吃": 2, "壽": 3, "司": 4}
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
        self.Wh_f = np.random.randn(hidden_dim, hidden_dim) & 0.01
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
            h_prev = self.step(x_seq[t:t+1], h_prev, self.Wx_b, self.Wx_b, self.b_b)

        return np.concatenate([h_f, h_b], axis=1)
    
class Classifier:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def forward(self, h_seq):
        z = h_seq @self.W + self.b
        return 1 / (1 + np.exp(-z))
    
def binary_cross_entropy(preds, targets):
    eps = 1e-7
    preds = np.clip(preds, eps, 1 - eps)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

def extract_keywor(tokens , scores, threshold=0.5):
    return [tok for tok, s in zip(tokens, scores) if s > threshold]