import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to create a generic network
def create_network(input_size, name):
    model = Sequential(name=name)
    model.add(Dense(64, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate synthetic data for demonstration
def generate_data(samples=1000, features=100):
    X = np.random.randn(samples, features)
    y = np.random.randint(0, 2, (samples, 1))
    return X, y

# Plotting function
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Generate and split the data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and train the individual networks
networks = {}
for i in range(4):
    networks[f'network{i}'] = create_network(X_train.shape[1], f'network{i}')
    history = networks[f'network{i}'].fit(X_train, y_train, epochs=10, batch_size=32, 
                                          validation_split=0.5, verbose=0)
    plot_history(history, f'Network {i}')

# Combine the outputs into a final network
inputs = [Input(shape=(X_train.shape[1],)) for _ in range(4)]
outputs = [networks[f'network{i}'](inputs[i]) for i in range(4)]
combined = Concatenate()(outputs)
final_layer = Dense(64, activation='relu')(combined)
final_layer = Dropout(0.2)(final_layer)
final_output = Dense(1, activation='sigmoid')(final_layer)

final_network = Model(inputs=inputs, outputs=final_output)
final_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the final network
final_history = final_network.fit([X_train, X_train, X_train, X_train], y_train, 
                                  epochs=10, batch_size=32, validation_split=0.5, verbose=0)
plot_history(final_history, 'Final Network')
