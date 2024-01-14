import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to generate dummy data
def generate_data(n_samples, n_features):
    data = np.random.normal(size=(n_samples, n_features))
    labels = np.random.randint(0, 2, size=n_samples)  # 0 for background, 1 for signal
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

# Function to create the combined network
def create_combined_network(networks, combined_input_shape):
    combined_input = tf.keras.Input(shape=combined_input_shape)
    concatenated = tf.keras.layers.concatenate([network(combined_input) for network in networks])
    x = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    x = tf.keras.layers.Dropout(0.2)(x)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    combined_model = tf.keras.Model(inputs=combined_input, outputs=final_output)
    combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return combined_model

# Generate dummy data
n_samples = 1000
n_features = 10
X, y = generate_data(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train individual networks
network1 = create_network(input_shape=(n_features,), name="network1")
network2 = create_network(input_shape=(n_features,), name="network2")
network3 = create_network(input_shape=(n_features,), name="network3")
network4 = create_network(input_shape=(n_features,), name="network4")

network1.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)
network2.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)
network3.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)
network4.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Create and train the combined model
combined_model = create_combined_network([network1, network2, network3, network4], combined_input_shape=(n_features,))
combined_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.25)

# Evaluate the combined model
loss, accuracy = combined_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Extract DNN scores
dnn_scores = combined_model.predict(X_test)
print("DNN Scores:", dnn_scores)
