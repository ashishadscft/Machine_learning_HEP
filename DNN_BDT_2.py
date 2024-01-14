import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to generate dummy data
def generate_data(n_samples, n_features, signal_ratio=0.5):
    data = np.random.normal(size=(n_samples, n_features))
    labels = np.random.choice([0, 1], size=n_samples, p=[1-signal_ratio, signal_ratio])
    return data, labels

# Function to create a DNN model
def create_network(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate dummy data
n_samples = 10000
n_features = 10
X, y = generate_data(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the network
network = create_network(input_shape=(n_features,), name="network")
history = network.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict DNN scores for the test data and separate into signal and background
dnn_scores = network.predict(X_test)
dnn_scores_signal = dnn_scores[y_test == 1].flatten()
dnn_scores_background = dnn_scores[y_test == 0].flatten()

# Print a sample of DNN scores for signal and background
print("Sample DNN Scores for Signal: ", dnn_scores_signal[:10])
print("Sample DNN Scores for Background: ", dnn_scores_background[:10])

# Define bins for the histograms and plot
bins = np.linspace(0, 1, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(10, 8))
plt.hist(dnn_scores_signal, bins=bins, histtype='step', color='black', label='Signal (Data)')
plt.hist(dnn_scores_background, bins=bins, histtype='step', color='blue', label='Background')
plt.xlabel('DNN Output')
plt.ylabel('Events')
plt.title('DNN Output Distribution')
plt.legend()
plt.show()

# Calculate and plot Data/Background Ratio
plt.figure(figsize=(10, 4))
hist_signal, _ = np.histogram(dnn_scores_signal, bins=bins)
hist_background, _ = np.histogram(dnn_scores_background, bins=bins)

# Safe division for the ratio
ratio = np.divide(hist_signal, hist_background, out=np.zeros_like(hist_signal, dtype=float), where=hist_background != 0)

# Calculate errors and handle division by zero
signal_errors = np.sqrt(hist_signal)
background_errors = np.sqrt(hist_background)

# Avoid division by zero in error calculation
with np.errstate(divide='ignore', invalid='ignore'):
    ratio_errors = np.abs(ratio) * np.sqrt((signal_errors / hist_signal)**2 + (background_errors / hist_background)**2)

# Replace NaNs and infs in ratio_errors with 0 (where background is 0)
ratio_errors[np.isnan(ratio_errors) | np.isinf(ratio_errors)] = 0

plt.errorbar(bin_centers, ratio, yerr=ratio_errors, fmt='o', color='green', label='Data/Background Ratio')
plt.xlabel('DNN Output')
plt.ylabel('Data / Background')
plt.title('Data/Background Ratio vs. DNN Output')
plt.axhline(y=1, color='gray', linestyle='--')
plt.legend()
plt.show()
