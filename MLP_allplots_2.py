import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

# Load and Prepare the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

# Build the MLP Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Custom Callback for Learning Rate Recording
#class LRRecorder(Callback):
 #   def on_train_begin(self, logs={}):
  #      self.lrs = []
   #     self.losses = []

    #def on_batch_end(self, batch, logs={}):
     #   self.lrs.append(self.model.optimizer.lr.numpy())
      #  self.losses.append(logs.get('loss'))

class LRRecorder(Callback):
    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs.get('loss'))
        
# Train the Model
epochs = 50
lr_recorder = LRRecorder()

history = model.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_test, y_test),
                    callbacks=[lr_recorder])

# Plot Loss vs. Learning Rate
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(lr_recorder.lrs, lr_recorder.losses)
plt.title('Loss vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')

# Plot Learning Rate vs. Epoch
plt.subplot(1, 3, 2)
plt.plot(range(epochs), lr_recorder.lrs)
plt.title('Learning Rate vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

# Plot Loss vs. Epoch
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

