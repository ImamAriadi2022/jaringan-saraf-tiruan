import numpy as np

# Fungsi aktivasi sigmoid dan turunannya
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Parameter jaringan
learning_rate = 0.2  # Laju pembelajaran
input_layer_size = 2  # Jumlah neuron di lapisan input
hidden_layer_size = 3  # Jumlah neuron di lapisan tersembunyi
output_layer_size = 1  # Jumlah neuron di lapisan output

# Input dan target untuk pola pertama
X = np.array([[1, 1]])  # Input: x1=1, x2=1
target = np.array([[0]])  # Target output: t=0

# Inisialisasi bobot secara acak
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))

# Bias
bias_hidden = np.random.uniform(-1, 1, (1, hidden_layer_size))
bias_output = np.random.uniform(-1, 1, (1, output_layer_size))

# Proses forward pass
def forward_pass(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    return hidden_output, final_output

# Iterasi Backpropagation untuk pola pertama
for epoch in range(1):  # Satu iterasi saja sesuai permintaan
    # Forward pass
    hidden_output, final_output = forward_pass(X)
    
    # Hitung error di lapisan output
    output_error = target - final_output
    output_delta = output_error * sigmoid_derivative(final_output)
    
    # Backpropagate error ke lapisan tersembunyi
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    # Perbarui bobot dan bias
    weights_hidden_output += learning_rate * hidden_output.T.dot(output_delta)
    weights_input_hidden += learning_rate * X.T.dot(hidden_delta)
    
    bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    # Tampilkan hasil iterasi
    print("Epoch 1: ")
    print("Bobot Input-Hidden yang Diperbarui:\n", weights_input_hidden)
    print("Bobot Hidden-Output yang Diperbarui:\n", weights_hidden_output)
    print("Bias Hidden yang Diperbarui:\n", bias_hidden)
    print("Bias Output yang Diperbarui:\n", bias_output)
    print("Error Lapisan Output:\n", output_error)