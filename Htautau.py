import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.manifold import TSNE

# Define the primary Tau-Tau Regression Model (tauNN)
def create_tauNN(input_shape):
    inputs = Input(shape=input_shape, name='input_tauNN')
    x = Dense(128, activation='relu')(inputs)
    for _ in range(4):
        x = Dense(128, activation='relu')(x)
    regression_output = Dense(7, activation='linear', name='regression_output')(x)
    classification_output = Dense(3, activation='softmax', name='classification_output')(x)
    return Model(inputs=inputs, outputs=[regression_output, classification_output], name='tauNN')

# Define the secondary DNN with skip-connections (pDNN)
def create_pDNN(input_shape):
    inputs = Input(shape=input_shape, name='input_pDNN')
    x = inputs
    # Create DNN layers with skip-connections
    for _ in range(3):  # Example: 3 layers, add more or fewer layers as needed
        y = Dense(128, activation='relu')(x)
        x = Add()([x, y])  # Skip-connection from input of the layer to the output
    output = Dense(2, activation='softmax', name='secondary_output')(x)
    return Model(inputs=inputs, outputs=output, name='pDNN')

# Define combined model function
def create_combined_model(tauNN_model, pDNN_model):
    # Connect the tauNN outputs to the pDNN inputs
    combined_input = tauNN_model.input
    tauNN_outputs = tauNN_model(combined_input)
    pDNN_input = Concatenate()([tauNN_outputs[0], tauNN_outputs[1]])  # Combine regression and classification outputs
    combined_output = pDNN_model(pDNN_input)
    return Model(inputs=combined_input, outputs=combined_output, name='combined_model')

# Placeholder for the number of input features for tauNN
tauNN_input_features = 100  # Replace with your actual number of input features

# Create both models
tauNN_model = create_tauNN((tauNN_input_features,))
pDNN_model = create_pDNN((10,))  # 10 combined outputs from tauNN (7 regression + 3 classification)

# Create the combined model
combined_model = create_combined_model(tauNN_model, pDNN_model)

# Compile the combined model
combined_model.compile(optimizer=Adam(learning_rate=1e-3),
                       loss='categorical_crossentropy',  # or appropriate loss function
                       metrics=['accuracy'])

# Model summary
combined_model.summary()

# Placeholder for model training (replace with actual data)
# history = combined_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32)

# For demonstration, simulate some training history data
history = {
    'loss': [1.0, 0.8, 0.6],
    'accuracy': [0.5, 0.6, 0.7],
    'val_loss': [1.1, 0.9, 0.7],
    'val_accuracy': [0.4, 0.5, 0.6]
}

# Plot the training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Placeholder for learning rate schedule (this needs to be defined based on your requirements)
def learning_rate_scheduler(epoch, lr):
    # Define your learning rate schedule strategy here
    return lr * tf.math.exp(-0.1)

# Callback to store learning rate and losses at each epoch
class LRSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LRSchedulerCallback, self).__init__()
        self.lr_rates = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.lr_rates.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs['loss'])

# Callback for per-layer analysis
per_layer_stats = {}

def log_layer_stats(layer):
    # Log weight statistics for each layer
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    per_layer_stats[layer.name] = {
        'weights': weights,
        'biases': biases,
        'weight_mean': weights.mean(),
        'weight_std': weights.std(),
        'bias_mean': biases.mean(),
        'bias_std': biases.std()
    }

layer_logging_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: [log_layer_stats(layer) for layer in combined_model.layers if len(layer.get_weights()) > 0])

# Include these callbacks in your model.fit call
callbacks = [
    LearningRateScheduler(learning_rate_scheduler),
    LRSchedulerCallback(),
    layer_logging_callback
]

# Placeholder for model training (replace with actual data)
# history = combined_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32, callbacks=callbacks)

# Assuming the model has been trained and LRSchedulerCallback has been used
# lr_scheduler_cb = LRSchedulerCallback()

# Plot Loss vs. Learning Rate
plt.figure(figsize=(6, 4))
plt.plot(lr_scheduler_cb.lr_rates, lr_scheduler_cb.losses)
plt.title('Loss vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xscale('log')  # Often you would plot this on a log scale
plt.show()

# Per-layer Weight and Bias Distributions
for layer_name, stats in per_layer_stats.items():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(stats['weights'].flatten(), bins=50)
    plt.title(f'{layer_name} - Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(stats['biases'].flatten(), bins=50)
    plt.title(f'{layer_name} - Bias Distribution')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.show()

# Assume 'y_val' is your validation targets and 'combined_model' is your trained model
y_pred = combined_model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(Y_val, axis=1)

# Confusion matrix
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# This requires a custom training loop or a callback to access gradients

# Here's a simplified version using a custom training step
def get_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = model.compiled_loss(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

# Collect gradients (just an example, replace with your actual training loop)
gradients = get_gradients(combined_model, X_train[:1], Y_train[:1])

# Plot histograms
for gradient, variable in zip(gradients, combined_model.trainable_variables):
    plt.hist(gradient.numpy().flatten(), bins=50)
    plt.title(f'Gradient Histogram for {variable.name}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.show()

from sklearn.metrics import roc_curve, auc

# ROC Curve for class 1 (assuming binary classification)
fpr, tpr, _ = roc_curve(y_true_classes, y_pred[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_true_classes, y_pred[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Extracting intermediate layer outputs
intermediate_model = Model(inputs=combined_model.input,
                           outputs=[layer.output for layer in combined_model.layers if 'dense' in layer.name])

# Assume we're using the first sample from the validation set
intermediate_outputs = intermediate_model.predict(X_val[:1])

# Plotting activations
for layer_output in intermediate_outputs:
    plt.matshow(layer_output[0], cmap='viridis')
    plt.colorbar()
    plt.title('Layer Activation')
    plt.show()

# Using the last layer's outputs for t-SNE
last_layer_outputs = intermediate_outputs[-1]

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(last_layer_outputs)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_true_classes[:len(tsne_result)], cmap='viridis')
plt.colorbar()
plt.title('t-SNE projection of the last layer activations')
plt.show()

# Compute correlation matrix for the last layer's output
corr_matrix = np.corrcoef(last_layer_outputs.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Last Layer Activations')
plt.show()
