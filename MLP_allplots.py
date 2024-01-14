import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data - scale the pixel values from a range of 0-255 to 0-1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the MLP model
def create_mlp():
    model = Sequential()

    # Flatten the input data
    model.add(Flatten(input_shape=(28, 28)))

    # Add five fully connected layers with 128 units each
    for _ in range(5):
        model.add(Dense(128, activation='relu'))

    # Output layer with 10 units (one for each class)
    model.add(Dense(10, activation='softmax'))

    return model
mlp_model = create_mlp()

# Compile the model
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = mlp_model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict the values from the test dataset
y_pred = mlp_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1) 
y_true = np.argmax(y_test, axis=1) 

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Binarize the labels for ROC curve
y_test_bin = label_binarize(y_test, classes=[i for i in range(10)])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate precision and recall for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred[:, i])

# Plot the precision-recall curve for each class
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of class {0} (area = {1:0.2f})'
             ''.format(i, average_precision[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.title('Multi-class Precision-Recall curve')
plt.show()

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        print(f"Learning rate after epoch {epoch}: {current_lr}")

lr_logger = LearningRateLogger()
# Include this callback in model.fit: callbacks=[lr_logger]

for layer in mlp_model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
        plt.hist(weights[0].flatten(), bins=100)
        plt.title(f'Weight Distribution in Layer {layer.name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()

# Choose a sample from the dataset
sample = X_test[0:1]

# Create a model for each layer output
layer_outputs = [layer.output for layer in mlp_model.layers]
activation_model = Model(inputs=mlp_model.input, outputs=layer_outputs)

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model

# Assuming mlp_model is your trained model and X_test is your test dataset
# First, create a model that will return these activations, given the model input
layer_outputs = [layer.output for layer in mlp_model.layers] 
activation_model = Model(inputs=mlp_model.input, outputs=layer_outputs)

# Choose a sample to visualize
sample = X_test[0:1]  # Adjust this to pick a specific sample

# Now, use the model to predict and fetch the activations
activations = activation_model.predict(sample)

# Visualize the activations
for activation in activations:
    # Check if the activation is one-dimensional
    if len(activation.shape) == 2 and activation.shape[0] == 1:
        # Reshape to two dimensions if it's one-dimensional
        activation_reshaped = np.reshape(activation, (activation.shape[1], 1))
        plt.matshow(activation_reshaped, cmap='viridis')
        plt.colorbar()
    else:
        plt.matshow(activation, cmap='viridis')
        plt.colorbar()
    plt.show()


# Get the activations
#activations = activation_model.predict(sample)

# Plot the activations of each layer
#for i, activation in enumerate(activations):
 #   plt.matshow(activation[0, :], cmap='viridis')
  #  plt.title(f'Layer {i+1} Activation')
   # plt.colorbar()
   # plt.show()

# Assuming you have a reasonable number of neurons per layer
#activations_df = pd.DataFrame(activations[1].reshape(-1, 128))  # Example for the second layer

# Calculate the correlation matrix
#corr = activations_df.corr()

# Plot the heatmap
#sns.heatmap(corr, cmap='coolwarm', annot=False)
#plt.title('Layer Activation Correlations')
#plt.show()

for i, activation in enumerate(activations):
    plt.hist(activation[0], bins=30)
    plt.title(f'Histogram of Activations in Layer {i+1}')
    plt.xlabel('Activation')
    plt.ylabel('Frequency')
    plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Across Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# This requires storing the weights after each epoch
# Here's a simplified approach using a custom callback
class WeightHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, batch, logs={}):
        self.weights.append(self.model.get_weights())

weight_history = WeightHistory()

# Include this callback in model.fit: callbacks=[weight_history]

# After training, plot the weight changes for a specific layer
layer_index = 1  # Change this based on which layer you want to analyze
weights_layer = np.array([w[layer_index][0] for w in weight_history.weights])

for i in range(weights_layer.shape[1]):
    plt.plot(weights_layer[:, i])
plt.title(f'Weight Changes in Layer {layer_index+1} Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.show()

# Select the layer for visualization (e.g., the last but one layer)
activations_to_visualize = activations[-2]  # Assuming this is a 2D array

# T-SNE transformation
tsne = TSNE(n_components=2, random_state=0)
activations_tsne = tsne.fit_transform(activations_to_visualize)

# Plot
plt.scatter(activations_tsne[:, 0], activations_tsne[:, 1], c=y_true, cmap='viridis')
plt.colorbar()
plt.title('T-SNE Visualization of Layer Activations')
plt.show()

plt.hist(y_pred.max(axis=1), bins=50)
plt.title('Histogram of Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# Assuming a regression task
y_pred_regression = model.predict(X_test)  # Predicted values
plt.scatter(y_test, y_pred_regression)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Predictions vs Actuals')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.show()

# This is an example for a simple classifier with 2D data
# Assume 'classifier' is your trained model and 'data' is your dataset

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(data[:, 0], data[:, 1], c=target, s=20, edgecolor='k')
plt.title('Decision Boundary')
plt.show()

# Assuming a multi-class classification problem
accuracies = []
for class_id in range(num_classes):
    class_mask = y_true == class_id
    class_accuracy = accuracy_score(y_true[class_mask], y_pred_classes[class_mask])
    accuracies.append(class_accuracy)

plt.bar(range(num_classes), accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.show()

# Assuming y_pred and y_true are available
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Per-Class ROC Curves')
plt.legend(loc="lower right")
plt.show()

mcc = matthews_corrcoef(y_true, y_pred_classes)
plt.bar(['MCC'], [mcc])
plt.title('Matthews Correlation Coefficient')
plt.ylim([-1, 1])
plt.show()

# Assuming y_pred and y_true are available
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
    avg_precision = average_precision_score(y_true == i, y_pred[:, i])
    plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Per-Class Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

plt.bar(['Sensitivity', 'Specificity'], [sensitivity, specificity])
plt.title('Sensitivity and Specificity')
plt.ylim([0, 1])
plt.show()

incorrect_indices = np.where(y_pred_classes != y_true)[0]
incorrect_predictions = y_pred_classes[incorrect_indices]
true_labels = y_true[incorrect_indices]

plt.figure(figsize=(15, 5))
for i, index in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f'Predicted: {incorrect_predictions[i]}, True: {true_labels[i]}')
    plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred_classes))


from sklearn.metrics import multilabel_confusion_matrix

multi_conf_matrix = multilabel_confusion_matrix(y_true, y_pred_classes)
# You can then sum up the respective counts from the confusion matrices.

from sklearn.metrics import f1_score

f1_scores = [f1_score(y_true, y_pred_classes, labels=[i], average='weighted') for i in range(n_classes)]

plt.bar(range(n_classes), f1_scores)
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.title('F1 Scores for Each Class')
plt.show()

def lr_schedule(epoch):
    return 10 ** (-epoch)

callback = LearningRateScheduler(lr_schedule)
history = mlp_model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=[callback])

plt.plot(history.history['lr'], history.history['loss'])
plt.gca().set_xscale('log')
plt.title('Loss vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()

# Assuming you have a history object from training with a dynamic learning rate
plt.plot(history.epoch, history.history['lr'])
plt.title('Learning Rate over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

# Choose a sample from the dataset
sample = X_test[0].reshape(1, 28, 28)

# Create a model that will return these outputs, given the model input
layer_outputs = [layer.output for layer in mlp_model.layers] 
activation_model = Model(inputs=mlp_model.input, outputs=layer_outputs)

# Forward pass
activations = activation_model.predict(sample)

# Visualization of each layer's activation
for i, activation in enumerate(activations):
    plt.matshow(activation[0, :, :], cmap='viridis')
    plt.title(f'Layer {i + 1}')
    plt.show()

for layer in mlp_model.layers:
    if 'dense' in layer.name:
        weights = layer.get_weights()[0]
        plt.hist(weights.flatten(), bins=100)
        plt.title(f'Weight Histogram for {layer.name}')
        plt.xlabel('Weights')
        plt.ylabel('Frequency')
        plt.show()

# Assuming activations is a list of layer outputs
correlations = []
for i in range(len(activations)):
    for j in range(i+1, len(activations)):
        corr = np.corrcoef(activations[i].flatten(), activations[j].flatten())[0, 1]
        correlations.append((i, j, corr))

# Plotting
for i, j, corr in correlations:
    plt.matshow(corr, cmap='coolwarm')
    plt.title(f'Correlation between Layer {i + 1} and Layer {j + 1}')
    plt.colorbar()
    plt.show()

from sklearn.cluster import KMeans

# Choose a layer to analyze
layer_activations = activations[some_layer_index]

# Flatten the activations
flattened_activations = layer_activations.reshape(layer_activations.shape[0], -1)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(flattened_activations)
cluster_centers = kmeans.cluster_centers_

# Visualization
plt.scatter(flattened_activations[:, 0], flattened_activations[:, 1], c=kmeans.labels_)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
plt.title('Neuron Activation Clustering')
plt.show()


# Define a callback to record activations
class ActivationLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, layer_name, input_sample):
        self.model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.activations = []
        self.input_sample = input_sample

    def on_epoch_end(self, epoch, logs=None):
        activation = self.model.predict(self.input_sample)
        self.activations.append(activation)

# Use a sample for tracking
sample = X_test[0:1]
activation_logger = ActivationLogger(mlp_model, 'dense_1', sample)

# Train the model
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[activation_logger])

# Plotting the evolution of activations
for epoch_activation in activation_logger.activations:
    plt.matshow(epoch_activation[0, :, :], cmap='viridis')
    plt.title('Activations Over Epochs')
    plt.show()

# Choose a layer to analyze
layer_activations = activations[some_layer_index]

# Plotting output distribution for each neuron
for neuron_idx in range(layer_activations.shape[-1]):
    neuron_activations = layer_activations[:, :, neuron_idx]
    plt.hist(neuron_activations.flatten(), bins=50)
    plt.title(f'Distribution of outputs for neuron {neuron_idx} in layer {some_layer_index}')
    plt.xlabel('Activation')
    plt.ylabel('Frequency')
    plt.show()

# Choose a sample and a layer
sample = X_test[0:1]
layer_name = 'dense_1'

# Model for activations
activation_model = Model(inputs=mlp_model.input, outputs=mlp_model.get_layer(layer_name).output)

# Vary each feature slightly and observe the change in activations
for feature_idx in range(sample.shape[1]):
    modified_sample = np.copy(sample)
    modified_sample[0, feature_idx] += 0.1  # Slight modification
    activation = activation_model.predict(modified_sample)
    plt.plot(activation.flatten(), label=f'Feature {feature_idx}')
plt.title('Change in Activations with Feature Modification')
plt.legend()
plt.show()

for layer in mlp_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        gamma, beta = layer.get_weights()[:2]
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(gamma)
        plt.title(f'Gamma values of {layer.name}')

        plt.subplot(1, 2, 2)
        plt.hist(beta)
        plt.title(f'Beta values of {layer.name}')

        plt.show()

from sklearn.decomposition import PCA

# Assuming 'embedding_layer' is your embedding layer
embeddings = mlp_model.get_layer('embedding_layer').get_weights()[0]
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.title('PCA of Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Custom callback to record gradients
class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            inputs = self.model.input
            outputs = self.model.output
            grads = tape.gradient(outputs, self.model.trainable_variables)
            for i, grad in enumerate(grads):
                plt.hist(grad.numpy().flatten(), bins=50)
                plt.title(f'Gradient Histogram for Layer {i} at Epoch {epoch}')
                plt.xlabel('Gradient Value')
                plt.ylabel('Frequency')
                plt.show()

gradient_histogram_callback = GradientHistogramCallback()
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[gradient_histogram_callback])

# This is just a schematic example
def plot_attention_map(model, input_sample):
    attention_layer = model.get_layer('attention_layer')
    attention_output = attention_layer(input_sample)
    plt.matshow(attention_output)
    plt.title('Attention Map')
    plt.colorbar()
    plt.show()

# Example usage
plot_attention_map(transformer_model, some_input_sample)

dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []

for rate in dropout_rates:
    temp_model = create_mlp(dropout_rate=rate)
    temp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    temp_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = temp_model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)

plt.plot(dropout_rates, accuracies)
plt.title('Model Accuracy as a Function of Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy')
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the labels for multi-class ROC
y_test_binarized = label_binarize(y_true, classes=range(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.title('Multi-Class ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import silhouette_score

# Assuming 'cluster_layer' is the output of a clustering layer
cluster_labels = mlp_model.get_layer('cluster_layer').predict(X_test)
silhouette_avg = silhouette_score(X_test, cluster_labels)

print(f'Silhouette Score: {silhouette_avg}')

for layer in mlp_model.layers:
    if len(layer.get_weights()) > 0:
        weights = layer.get_weights()[0]
        sns.heatmap(weights, cmap='viridis')
        plt.title(f'Heatmap of Weights in {layer.name}')
        plt.show()

import pandas as pd

# Assuming 'X_train' and 'y_train' are your features and labels
df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
df['label'] = np.argmax(y_train, axis=1)

# Box plot for each feature by class
for column in df.columns[:-1]:  # Exclude the label column
    sns.boxplot(x='label', y=column, data=df)
    plt.title(f'Box Plot of Feature {column}')
    plt.show()

training_sizes = np.linspace(0.1, 1.0, 5)
train_errors, test_errors = [], []

for size in training_sizes:
    subset_size = int(size * X_train.shape[0])
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]

    mlp_model_subset = create_mlp()
    mlp_model_subset.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mlp_model_subset.fit(X_train_subset, y_train_subset, epochs=10, batch_size=32, verbose=0)

    train_loss, _ = mlp_model_subset.evaluate(X_train_subset, y_train_subset, verbose=0)
    test_loss, _ = mlp_model_subset.evaluate(X_test, y_test, verbose=0)

    train_errors.append(train_loss)
    test_errors.append(test_loss)

plt.plot(training_sizes, train_errors, label='Training error')
plt.plot(training_sizes, test_errors, label='Testing error')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Error')
plt.legend()
plt.show()


