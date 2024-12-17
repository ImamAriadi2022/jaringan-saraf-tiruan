import numpy as np
import pandas as pd

# Fungsi aktivasi sigmoid dan turunannya
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Transformasi data menggunakan rumus
# x' = 0.8 * ((x - a) / (b - a)) + 0.1
def transform_data(data, a, b):
    return 0.8 * (data - a) / (b - a) + 0.1

# Baca data dari file CSV
data = pd.read_csv('dataset.csv').values

# Ambil data nilai dan target
input_data = data[:, 1:5]  # 4 kolom input
output_data = data[:, -1]   # Kolom terakhir sebagai target

# Transformasi data input dan target
transformed_input = np.zeros_like(input_data, dtype=np.float32)
for i in range(input_data.shape[1]):
    a, b = np.min(input_data[:, i]), np.max(input_data[:, i])
    transformed_input[:, i] = transform_data(input_data[:, i], a, b)

output_min, output_max = np.min(output_data), np.max(output_data)
transformed_output = transform_data(output_data, output_min, output_max).reshape(-1, 1)

# Parameter JST
input_neurons = 4
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1

# Inisialisasi bobot dan bias secara random
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# Training JST menggunakan Backpropagation
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(transformed_input, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Hitung error
    output_error = transformed_output - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update bobot dan bias
    weights_hidden_output += learning_rate * hidden_output.T.dot(output_delta)
    weights_input_hidden += learning_rate * transformed_input.T.dot(hidden_delta)
    bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    # Cetak error tiap 1000 epoch
    if epoch % 1000 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
# Hasil bobot dan bias setelah training
print("\n=== Hasil Akhir Training ===")
print("Bobot Input-Hidden:\n", weights_input_hidden)
print("Bobot Hidden-Output:\n", weights_hidden_output)
print("Bias Hidden:\n", bias_hidden)
print("Bias Output:\n", bias_output)
    
    # Prediksi setelah training
hidden_input = np.dot(transformed_input, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
predictions = sigmoid(final_input)
    
# Kembalikan prediksi ke skala asli
final_predictions = (predictions - 0.1) * (output_max - output_min) / 0.8 + output_min
    
# Cetak hasil prediksi
print("\n=== Hasil Prediksi ===")
for i, prediction in enumerate(final_predictions):
    print(f"Data ke-{i+1}: Prediksi Nilai = {prediction[0]:.2f}")