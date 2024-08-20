import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

class MultiLayerLSTM:
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        self.num_layers = num_layers
        self.lstm_layers = []
        
        # Create multiple LSTM layers
        for _ in range(num_layers):
            self.lstm_layers.append(LSTMCell(input_size, hidden_size))
            input_size = hidden_size  # Update input size for subsequent layers
        
        # Output weights
        self.Why = np.random.randn(hidden_size, output_size)
        self.by = np.random.randn(output_size)
        
    def forward(self, x):
        for layer in self.lstm_layers:
            x = layer.forward(x)
        
        y = sigmoid(np.dot(x, self.Why) + self.by)
        return y

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
            
class LSTMCell:
    def __init__(self, input_size, hidden_size, dropout=0):
        # Gate weights
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size)
        self.Wg = np.random.randn(hidden_size, hidden_size + input_size)
        
        # Biases
        self.bf = np.random.randn(hidden_size)
        self.bi = np.random.randn(hidden_size)
        self.bo = np.random.randn(hidden_size)
        self.bg = np.random.randn(hidden_size)
        
        # Initial states
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
        self.dropout = dropout


    def forward(self, x):
        combined = np.hstack((x, self.h))
        
        # Apply dropout to input
        # 
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=combined.shape)
            combined *= mask / (1 - self.dropout)
        
        # Forget gate, input gate, output gate
        ft = sigmoid(np.dot(self.Wf, combined) + self.bf)
        it = sigmoid(np.dot(self.Wi, combined) + self.bi)
        ot = sigmoid(np.dot(self.Wo, combined) + self.bo)
        gt = np.tanh(np.dot(self.Wg, combined) + self.bg)
        
        self.c = ft * self.c + it * gt
        self.h = ot * np.tanh(self.c)
        
        return self.h

    def reset_state(self):
        self.h = np.zeros_like(self.h)
        self.c = np.zeros_like(self.c)