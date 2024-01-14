import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to generate dummy data
def generate_data(n_samples, n_features, signal_ratio=0.5):
    data = np.random.normal(size=(n_samples, n_features))
    labels = np.random.choice([0, 1], size=n_samples, p=[1-signal_ratio, signal_ratio])  # Background: 0, Signal: 1
    return data, labels

# Function to create a DNN model
def create_network(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate dummy data
n_samples = 10000
n_features = 10
X, y = generate_data(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the network
network = create_network(input_shape=(n_features,), name="network")
network.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Predict DNN scores for the test data
dnn_scores = network.predict(X_test)

# Separate the scores into signal and background
dnn_scores_signal = dnn_scores[y_test == 1].flatten()
dnn_scores_background = dnn_scores[y_test == 0].flatten()

# Define bins for the histograms
bins = np.linspace(0, 1, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot signal and background DNN scores
plt.figure(figsize=(10, 8))
plt.hist(dnn_scores_signal, bins=bins, histtype='step', color='black', label='Signal (Data)')
plt.hist(dnn_scores_background, bins=bins, histtype='step', color='blue', label='Background')
plt.xlabel('DNN Output')
plt.ylabel('Events')
plt.title('DNN Output Distribution')
plt.legend()
plt.show()

# Plot Data/Background Ratio
plt.figure(figsize=(10, 4))
hist_signal, _ = np.histogram(dnn_scores_signal, bins=bins)
hist_background, _ = np.histogram(dnn_scores_background, bins=bins)
ratio = hist_signal / hist_background
ratio_errors = ratio * np.sqrt((np.sqrt(hist_signal) / hist_signal)**2 + (np.sqrt(hist_background) / hist_background)**2)
plt.errorbar(bin_centers, ratio, yerr=ratio_errors, fmt='o', color='green', label='Data/Background Ratio')
plt.xlabel('DNN Output')
plt.ylabel('Data / Background')
plt.title('Data/Background Ratio vs. DNN Output')
plt.axhline(y=1, color='gray', linestyle='--')
plt.legend()
plt.show()
