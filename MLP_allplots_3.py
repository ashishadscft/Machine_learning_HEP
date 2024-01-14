import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt

# Load and prepare CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Custom callback for recording LR and losses
class LRRecorder(Callback):
    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs.get('loss'))

# Train the model
epochs = 20
lr_recorder = LRRecorder()
callback = LearningRateScheduler(scheduler)

history = model.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_test, y_test),
                    callbacks=[callback, lr_recorder])

# Plotting
plt.figure(figsize=(12, 4))

# Plot Loss vs. Learning Rate
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
