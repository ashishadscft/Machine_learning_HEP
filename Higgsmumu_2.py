import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from keras_tuner import RandomSearch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to create a network with hyperparameters
def create_network(hp, input_size, name):
    model = Sequential(name=name)
    model.add(Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(input_size,)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('units_3', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to build a model with hyperparameters (for keras-tuner)
def model_builder(hp):
    return create_network(hp, X_train.shape[1], "network_hp_tuned")

# Function to generate synthetic data
def generate_data(samples=1000, features=100):
    X = np.random.randn(samples, features)
    y = np.random.randint(0, 2, (samples, 1))
    return X, y

# Function to plot training history
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Generate and split the data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train and tune four individual networks
#networks = {}
#for i in range(4):
 #   tuner = RandomSearch(model_builder,
#                         objective='val_accuracy',
 #                        max_trials=5,
  #                       directory=f'keras_tuner_dir_{i}',
   #                      project_name=f'network_tuning_{i}')

   # tuner.search(X_train, y_train, epochs=10, validation_split=0.5, verbose=1)
   # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
   # model = tuner.hypermodel.build(best_hps)
   # history = model.fit(X_train, y_train, epochs=10, validation_split=0.5, verbose=1)
   # networks[f'network{i}'] = model
   # plot_history(history, f'Network {i}')


# Train and tune four individual networks
networks = {}
for i in range(4):
    # Modified to build a model with a unique name for each network
    def model_builder(hp):
        return create_network(hp, X_train.shape[1], f'network{i}')

    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=5,
        directory=f'keras_tuner_dir_{i}',
        project_name=f'network_tuning_{i}'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.5, verbose=1)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.5, verbose=1)
    networks[f'network{i}'] = model
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
final_history = final_network.fit([X_train, X_train, X_train, X_train], y_train, epochs=10, batch_size=32, validation_split=0.5, verbose=1)
plot_history(final_history, 'Final Network')
