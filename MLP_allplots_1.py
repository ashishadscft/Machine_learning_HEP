import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.datasets import fashion_mnist

# Load the MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the MLP model
def create_mlp():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        *[Dense(128, activation='relu') for _ in range(5)],
        Dense(10, activation='softmax')
    ])
    return model

# Learning Rate Finder Callback
class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-5, max_lr=1e-1, steps_per_epoch=None, epochs=None):
        super(LRFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('loss', []).append(logs.get('loss'))
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

# Gradient Histogram Callback
class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(GradientHistogramCallback, self).__init__()
        self.model = model
        # Get the actual loss function from its string name
        self.loss_fn = CategoricalCrossentropy(from_logits=False)

    def on_epoch_end(self, epoch, logs=None):
        # Take a small subset of the training data to compute the gradients
        subset_size = min(1024, len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]

        gradients = []
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel
                with tf.GradientTape() as tape:
                    tape.watch(weights)
                    predictions = self.model(X_subset, training=True)
                    loss = self.loss_fn(y_subset, predictions)

                grads = tape.gradient(loss, weights)
                gradients.append(grads)

        # Plotting gradient histograms
        for i, gradient in enumerate(gradients):
            if gradient is not None:
                plt.hist(gradient.numpy().flatten(), bins=50)
                plt.title(f'Layer {i+1} Gradient Histogram at Epoch {epoch}')
                plt.xlabel('Gradient Value')
                plt.ylabel('Frequency')
                plt.show()

# Find optimal learning rate
mlp_model = create_mlp()
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lr_finder = LRFinder(min_lr=1e-5, max_lr=1, steps_per_epoch=len(X_train) // 32, epochs=5)
mlp_model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=[lr_finder])

# Plot the loss vs. learning rate
plt.figure(figsize=(10, 6))
plt.plot(lr_finder.history['lr'], lr_finder.history['loss'])
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Loss vs. Learning Rate')
plt.show()

# Select a learning rate based on the plot and retrain the model
selected_lr = 1e-3  # Adjust based on the plot
mlp_model = create_mlp()
mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=selected_lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Retrain the model with standard metrics and GradientHistogramCallback
history = mlp_model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split=0.2,
                        callbacks=[GradientHistogramCallback(mlp_model)])

# Evaluate the model
loss, accuracy = mlp_model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.5f}%")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
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

# Compute and plot the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
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
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate precision and recall for each class
precision, recall, average_precision = {}, {}, {}
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred[:, i])

# Plot the precision-recall curve for each class
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.title('Multi-class Precision-Recall curve')
plt.show()

# Layer weight distribution
for layer in mlp_model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
        plt.hist(weights[0].flatten(), bins=100)
        plt.title(f'Weight Distribution in Layer {layer.name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()

# Activation visualizations
layer_outputs = [layer.output for layer in mlp_model.layers] 
activation_model = Model(inputs=mlp_model.input, outputs=layer_outputs)
sample = X_test[0:1]  # Using the first test sample
activations = activation_model.predict(sample)

for i, activation in enumerate(activations):
    if len(activation.shape) == 2 and activation.shape[0] == 1:
        activation_reshaped = np.reshape(activation, (activation.shape[1], 1))
        plt.matshow(activation_reshaped, cmap='viridis')
    else:
        plt.matshow(activation, cmap='viridis')
    plt.colorbar()
    plt.title(f'Layer {i+1} Activation')
    plt.show()

# Select the layer to visualize
selected_layer_activations = activations[-2]  # Example: second last layer

# Ensure you have enough samples for t-SNE
n_samples = selected_layer_activations.shape[0]
if n_samples > 1:
    perplexity_value = min(30, n_samples - 1)  # Set perplexity to a valid value
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
    activations_tsne = tsne.fit_transform(selected_layer_activations)

    plt.scatter(activations_tsne[:, 0], activations_tsne[:, 1], c=y_true, cmap='viridis')
    plt.colorbar()
    plt.title('T-SNE Visualization of Layer Activations')
    plt.show()
else:
    print("Not enough samples for t-SNE visualization.")

class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gradients = []
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                gradient = self.get_gradients_of_trainable_weights(layer.kernel)
                gradients.append(gradient)
        # Plotting gradient histograms
        for i, gradient in enumerate(gradients):
            plt.hist(gradient, bins=50)
            plt.title(f'Layer {i+1} Gradient Histogram at Epoch {epoch}')
            plt.xlabel('Gradient Value')
            plt.ylabel('Frequency')
            plt.show()

    def get_gradients_of_trainable_weights(self, weights):
        with tf.GradientTape() as tape:
            _ = self.model(X_train, training=True)  # Forward pass
            loss = self.model.total_loss
        grads = tape.gradient(loss, weights)
        return grads.numpy().flatten()

gradient_histogram_callback = GradientHistogramCallback()

per_class_accuracies = []

for class_id in range(10):  # Assuming 10 classes for MNIST
    class_mask = y_true == class_id
    class_accuracy = accuracy_score(y_true[class_mask], y_pred_classes[class_mask])
    per_class_accuracies.append(class_accuracy)

plt.bar(range(10), per_class_accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.show()

# Convert predictions and true labels to binary format for MCC calculation
y_pred_binary = label_binarize(y_pred_classes, classes=[i for i in range(10)])
y_true_binary = label_binarize(y_true, classes=[i for i in range(10)])

mcc = matthews_corrcoef(y_true_binary.argmax(axis=1), y_pred_binary.argmax(axis=1))
plt.bar(['MCC'], [mcc])
plt.title('Matthews Correlation Coefficient')
plt.ylim([-1, 1])
plt.show()

incorrect_indices = np.where(y_pred_classes != y_true)[0]

plt.figure(figsize=(15, 5))
for i, index in enumerate(incorrect_indices[:10]):  # Display first 10 incorrect predictions
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f'Pred: {y_pred_classes[index]}, True: {y_true[index]}')
    plt.tight_layout()
plt.show()

def lr_schedule(epoch):
    # Example of a simple decay schedule
    initial_lr = 0.001  # Starting learning rate
    decay = 0.1         # Decay factor
    new_lr = initial_lr * (1 / (1 + decay * epoch))
    return new_lr

# Assuming a learning rate schedule was used during training
plt.plot(history.epoch, [lr_schedule(epoch) for epoch in history.epoch])
plt.title('Learning Rate over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

# Select a layer to visualize
selected_layer_index = 2  # Example: the third layer in the model
selected_layer_activations = activations[selected_layer_index]

# Check if there are enough samples and features
if selected_layer_activations.ndim == 2 and selected_layer_activations.shape[0] > 1:
    n_neurons = selected_layer_activations.shape[1]

    plt.figure(figsize=(15, 5))
    for i in range(min(10, n_neurons)):  # Display first 10 neurons (or fewer if less than 10)
        plt.subplot(2, 5, i + 1)
        plt.plot(selected_layer_activations[:, i])
        plt.title(f'Neuron {i}')
    plt.show()
else:
    print("Not enough data or incorrect dimensions for neuron activation visualization.")

# Select a layer for analysis
#selected_layer_index = 2  # Example: the third layer in the model
#selected_layer_activations = activations[selected_layer_index]

#plt.figure(figsize=(15, 5))
#for i in range(min(10, selected_layer_activations.shape[-1])):  # Display activations of first 10 neurons
 #   plt.subplot(2, 5, i + 1)
  #  plt.plot(selected_layer_activations[0, :, i])
   # plt.title(f'Neuron {i}')
#plt.show()

# Compute correlation matrix for a selected layer
layer_activations = activations[selected_layer_index]
flattened_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
correlation_matrix = np.corrcoef(flattened_activations.T)

sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Layer Activations')
plt.show()

# Ensure you're accessing the weights of the first Dense layer
first_dense_layer_index = 1  # Adjust this based on your model's architecture
if len(mlp_model.layers[first_dense_layer_index].get_weights()) > 0:
    first_layer_weights = mlp_model.layers[first_dense_layer_index].get_weights()[0]

    # Calculate feature importance as the mean of absolute weights
    feature_importance = np.mean(np.abs(first_layer_weights), axis=1)

    # Plot the feature importance
    plt.bar(range(feature_importance.shape[0]), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance (Mean Absolute Weight)')
    plt.show()
else:
    print("The selected layer does not have weights or is not a Dense layer.")

# Pseudocode - Replace with your hyperparameter tuning results
hyperparameters = ['0.01', '0.001', '0.0001']  # Example: Learning rates
performance = [0.85, 0.9, 0.87]  # Example: Validation accuracies

plt.plot(hyperparameters, performance)
plt.title('Model Performance for Different Hyperparameters')
plt.xlabel('Hyperparameter Value')
plt.ylabel('Model Performance Metric')
plt.show()

# Pseudocode - You need to run the model with different dropout rates and capture performance
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
performance = [0.88, 0.89, 0.87, 0.86, 0.85]

plt.plot(dropout_rates, performance)
plt.title('Impact of Dropout Rate on Model Performance')
plt.xlabel('Dropout Rate')
plt.ylabel('Performance Metric')
plt.show()

# Pseudocode - Example for models with probabilistic outputs
prediction_probabilities = mlp_model.predict(X_test)
prediction_uncertainty = np.std(prediction_probabilities, axis=1)

plt.hist(prediction_uncertainty, bins=50)
plt.title('Histogram of Prediction Uncertainties')
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.show()

# Pseudocode - Replace with your cross-validation results
cross_val_scores = [0.88, 0.90, 0.89, 0.87, 0.91]  # Example scores

plt.plot(range(1, len(cross_val_scores) + 1), cross_val_scores)
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Assuming you have the necessary data and model defined as in your original code

# Neuron Activation Distributions
def plot_neuron_activations(model, X_sample, layer_index=0, neuron_indices=[0, 1]):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_index].output)
    intermediate_output = intermediate_layer_model.predict(X_sample)

    for i in neuron_indices:
        plt.hist(intermediate_output[:, i], bins=50)
        plt.title(f'Distribution of Activations for Neuron {i} in Layer {layer_index}')
        plt.xlabel('Activation')
        plt.ylabel('Frequency')
        plt.show()

# Use a small sample of your data for this plot
# X_sample = X_test[:100]  # for example
# plot_neuron_activations(mlp_model, X_sample)

# Weight Distribution over Epochs (to be implemented as a callback)
# This requires storing weight histograms in a callback during training


# Class Prediction Error Rates
def plot_class_prediction_errors(y_true, y_pred_classes, num_classes=10):
    error_rates = []
    for i in range(num_classes):
        class_errors = np.sum((y_pred_classes == i) & (y_true != i))
        total = np.sum(y_true == i)
        error_rates.append(class_errors / total)

    plt.bar(range(num_classes), error_rates)
    plt.title('Class Prediction Error Rates')
    plt.xlabel('Class')
    plt.ylabel('Error Rate')
    plt.show()

# Layer Output Dimensionality Reduction (e.g., PCA)
def plot_layer_output_pca(model, X_sample, layer_index=0):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_index].output)
    intermediate_output = intermediate_layer_model.predict(X_sample)
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(intermediate_output)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title(f'PCA of Layer {layer_index} Outputs')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()





























