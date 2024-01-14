import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE

def clear_plot_directory(directory):
    # Use a glob pattern to find all .png files in the directory
    files = glob.glob(os.path.join(directory, '*.png'))

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error: {f} : {e.strerror}")

# Call the function at the start of your script
plot_directory = '/Users/ashishsehrawat/DNN_training/'
clear_plot_directory(plot_directory)

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Flatten the images
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

# Build a simple DNN
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the accuracy
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
#plt.show()
plt.savefig('/Users/ashishsehrawat/DNN_training/'+'model_accuracy_loss.png')
plt.close()

# Predictions for test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# ROC Curve and AUC for each class
n_classes = y_test.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve for each class
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for each class')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('/Users/ashishsehrawat/DNN_training/'+'roc_curve.png')
plt.close()

# Precision-Recall Curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=f'class {i}')

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
#plt.show()
plt.savefig('/Users/ashishsehrawat/DNN_training/'+'precision_recall_curve.png')
plt.close()

# Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.show()
plt.savefig('/Users/ashishsehrawat/DNN_training/'+'confusion_matrix.png')
plt.close()

# Histograms of Weights and Biases
for layer in model.layers:
    if 'dense' in layer.name:
        weights, biases = layer.get_weights()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(weights.flatten(), bins=50)
        plt.title(f'Weights Histogram - {layer.name}')
        plt.subplot(1, 2, 2)
        plt.hist(biases.flatten(), bins=50)
        plt.title(f'Biases Histogram - {layer.name}')
        #plt.show()
        plt.savefig(f'/Users/ashishsehrawat/DNN_training/histogram_{layer.name}.png')
        plt.close()
        
# Gradient Norms (requires a batch of data)
with tf.GradientTape() as tape:
    inputs = tf.convert_to_tensor(x_train[:32], dtype=tf.float32)
    true_outputs = tf.convert_to_tensor(y_train[:32], dtype=tf.float32)
    predictions = model(inputs)
    loss = tf.keras.losses.categorical_crossentropy(true_outputs, predictions, from_logits=False)
grads = tape.gradient(loss, model.trainable_weights)
grad_norms = [np.linalg.norm(g.numpy().flatten()) for g in grads]

plt.figure(figsize=(8, 4))
plt.bar(range(len(grad_norms)), grad_norms)
plt.title('Gradient Norms')
plt.xlabel('Layer')
plt.ylabel('Norm')
#plt.show()
plt.savefig('/Users/ashishsehrawat/DNN_training/'+'gradient_norms.png')
plt.close()

# T-SNE Visualization of Embeddings
layer_outputs = [layer.output for layer in model.layers if 'dense' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train[:1000])  # Using subset for faster computation

for i, activation in enumerate(activations):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(activation)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.argmax(y_train[:1000], axis=1), cmap='viridis')
    plt.colorbar()
    plt.title(f'T-SNE of Layer {i+1} Activations')
    #plt.show()
    plt.savefig(f'/Users/ashishsehrawat/DNN_training/tsne_layer_{i+1}.png')
    plt.close()
