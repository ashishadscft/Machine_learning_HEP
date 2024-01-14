import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Define the primary Tau-Tau Regression Model (tauNN)
def create_tauNN(input_shape):
    inputs = Input(shape=input_shape, name='input_tauNN')
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    for _ in range(4):
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    regression_output = Dense(7, activation='linear', name='regression_output')(x)
    classification_output = Dense(3, activation='softmax', name='classification_output')(x)
    return Model(inputs=inputs, outputs=[regression_output, classification_output], name='tauNN')

# Define the secondary DNN with skip-connections (pDNN)
def create_pDNN(input_shape):
    inputs = Input(shape=input_shape, name='input_pDNN')
    x = inputs
    for _ in range(3):
        y = Dense(128, activation='relu')(x)
        x = Add()([x, y])
    output = Dense(2, activation='softmax', name='secondary_output')(x)
    return Model(inputs=inputs, outputs=output, name='pDNN')

# Define combined model function
def create_combined_model(tauNN_model, pDNN_model):
    combined_input = tauNN_model.input
    tauNN_outputs = tauNN_model(combined_input)
    pDNN_input = Concatenate()([tauNN_outputs[0], tauNN_outputs[1]])
    combined_output = pDNN_model(pDNN_input)
    return Model(inputs=combined_input, outputs=combined_output, name='combined_model')

# Create both models
tauNN_input_features = 100  # Replace with your actual number of input features
tauNN_model = create_tauNN((tauNN_input_features,))
pDNN_model = create_pDNN((10,))  # 10 combined outputs from tauNN (7 regression + 3 classification)
combined_model = create_combined_model(tauNN_model, pDNN_model)

# Compile the combined model
combined_model.compile(optimizer=Adam(learning_rate=1e-3),
                       loss={'regression_output': 'mean_squared_error', 'classification_output': 'categorical_crossentropy'},
                       metrics={'regression_output': 'mean_squared_error', 'classification_output': 'accuracy'})

# Learning rate scheduler
def learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Callbacks for learning rate logging and per-layer analysis
class LRSchedulerCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        tf.print("Learning rate:", lr)

lr_scheduler_cb = LearningRateScheduler(learning_rate_scheduler)
per_layer_stats = {}

def log_layer_stats(epoch, logs):
    for layer in combined_model.layers:
        if hasattr(layer, 'kernel'):
            weights, biases = layer.get_weights()
            per_layer_stats[layer.name] = {
                'weights': weights,
                'biases': biases
            }

layer_logging_callback = LambdaCallback(on_epoch_end=log_layer_stats)

# Train the model (replace with your actual data)
# history = combined_model.fit(X_train, [Y_train_reg, Y_train_class],
#                              validation_data=(X_val, [Y_val_reg, Y_val_class]),
#                              epochs=100,
#                              batch_size=32,
#                              callbacks=[lr_scheduler_cb, layer_logging_callback])

# After training, generate evaluation plots

# Confusion Matrix for classification output
y_pred_class = combined_model.predict(X_val)[1]  # Assuming index 1 is the classification output
y_pred_classes = np.argmax(y_pred_class, axis=1)
y_true_classes = np.argmax(Y_val_class, axis=1)  # Assuming Y_val_class is the true classification labels
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve and AUC for classification output
fpr, tpr, _ = roc_curve(y_true_classes, y_pred_class[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for classification output
precision, recall, _ = precision_recall_curve(y_true_classes, y_pred_class[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# t-SNE Visualization of the regression output
y_pred_reg = combined_model.predict(X_val)[0]  # Assuming index 0 is the regression output
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(y_pred_reg)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_true_classes, cmap='viridis')
plt.colorbar()
plt.title('t-SNE projection of the regression output')
plt.show()

# Correlation Matrix of the regression output
corr_matrix = np.corrcoef(y_pred_reg.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Regression Output')
plt.show()

# Assuming `X_test` is your test dataset
# This would get the softmax probability outputs for the classification task
probabilities = combined_model.predict(X_test)[1]  # [1] if it's the second output of your model

# Now, let's plot the stacked histograms
# 'probabilities' is assumed to be a NumPy array where each row is an instance
# and each column represents the probability of the corresponding class
# 'class_labels' should correspond to the names of the classes in the order they are output by the model

# Define class labels based on your model's output
class_labels = ['TT', 'DY', 'ST', 'W', 'WW', 'tth', 'EMK', 'SM H', 'VH']  # adjust these based on your classes

# Determine the number of bins for histogram
num_bins = 50

# Generate the histogram data
# We'll collect the bin edges and counts for each class
hist_data = [np.histogram(probabilities[:, i], bins=num_bins, range=(0,1)) for i in range(probabilities.shape[1])]

# The hist_data list contains pairs of (hist_counts, bin_edges) for each class
# We will use the bin_edges from the first class to plot all histograms
bin_edges = hist_data[0][1]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the stacked histograms
plt.figure(figsize=(12, 6))
bottom = np.zeros(num_bins)

for i, (hist_counts, _) in enumerate(hist_data):
    plt.bar(bin_centers, hist_counts, bottom=bottom, width=(bin_edges[1] - bin_edges[0]), label=class_labels[i])
    bottom += hist_counts

plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('DNN Out')
plt.ylabel('Count (log scale)')
plt.title('DNN Signal Category Visualization')
plt.legend()
plt.show()


from sklearn.inspection import permutation_importance

# Assuming X_val and Y_val are your validation datasets
results = permutation_importance(combined_model, X_val, Y_val, scoring='accuracy')

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(X_val.shape[1]), results.importances_mean)
plt.title('Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

# Assuming 'lr_scheduler_cb' is your LearningRateSchedulerCallback instance
# and 'history' contains the training history

for layer in combined_model.layers:
    if len(layer.get_weights()) > 0:  # Check if the layer has trainable parameters
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.title(f'{layer.name} - Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(lr_scheduler_cb.lr_rates, label='Learning Rate')
        plt.title(f'{layer.name} - Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.show()


# Plotting activation distribution
for layer_output in intermediate_outputs:
    plt.figure(figsize=(12, 6))
    plt.hist(layer_output.flatten(), bins=50)
    plt.title(f'{layer.name} - Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.show()

class WeightChangeCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_weights = [layer.get_weights()[0] for layer in self.model.layers if len(layer.get_weights()) > 0]
        self.weight_history.append(current_weights)

weight_change_cb = WeightChangeCallback(combined_model)
# Add 'weight_change_cb' to your callbacks list in model.fit()

# After training, compare weight changes
initial_weights = weight_change_cb.weight_history[0]
for epoch, weights in enumerate(weight_change_cb.weight_history[1:]):
    plt.figure(figsize=(10, 4))
    for layer_idx, (initial, current) in enumerate(zip(initial_weights, weights)):
        weight_change = np.abs(current - initial)
        plt.hist(weight_change.flatten(), bins=50, alpha=0.5, label=f'Layer {layer_idx}')
    plt.title(f'Weight Change Distribution at Epoch {epoch + 1}')
    plt.xlabel('Weight Change')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Assuming 'y_pred' contains model predictions and 'Y_val' contains true labels
errors = y_pred - Y_val
plt.hist(errors, bins=50)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

from sklearn.metrics import precision_recall_curve

# Assuming binary classification and 'y_pred' contains prediction probabilities for the positive class
precision, recall, thresholds = precision_recall_curve(Y_val[:, 1], y_pred[:, 1])

plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.title('Precision and Recall for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# Identify some incorrect predictions
incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
num_samples_to_display = 5  # You can adjust this number

for i in incorrect_indices[:num_samples_to_display]:
    plt.figure(figsize=(10, 2))
    plt.plot(X_val[i], 'bo-')
    plt.title(f'Example {i}: True label={y_true_classes[i]} Predicted={y_pred_classes[i]}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Assuming you have stored gradients in a list during training
for epoch, gradients in enumerate(stored_gradients):
    plt.figure(figsize=(10, 4))
    for layer_idx, gradient in enumerate(gradients):
        plt.hist(gradient.flatten(), bins=50, alpha=0.5, label=f'Layer {layer_idx}')
    plt.title(f'Gradient Distribution at Epoch {epoch + 1}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Calculate performance metrics for each class
class_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
plt.bar(range(len(class_accuracy)), class_accuracy)
plt.title('Class-wise Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.show()

# Assuming 'model' is your CNN and 'layer_names' are the names of convolutional layers
for layer_name in layer_names:
    layer_output = model.get_layer(layer_name).output
    intermediate_model = Model(inputs=model.input, outputs=layer_output)
    feature_maps = intermediate_model.predict(X_sample)  # X_sample is a sample input

    # Plot the feature maps
    num_features = feature_maps.shape[-1]  # Number of features in the feature map
    size = feature_maps.shape

# Assuming 'model' is your CNN
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)  # Normalize filter values

        # Plot the first few filters
        n_filters = min(filters.shape[3], 6)  # Number of filters to display
        plt.figure(figsize=(15, 3))
        for i in range(n_filters):
            plt.subplot(1, n_filters, i + 1)
            plt.imshow(filters[:, :, :, i], cmap='viridis')
            plt.title(f'Filter {i + 1}')
            plt.axis('off')
        plt.suptitle(f'Filters in {layer.name}')
        plt.show()

# Create a model to extract the output of the last dense layer
last_dense_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)
last_dense_output = last_dense_layer_model.predict(X_sample)  # Replace 'X_sample' with your data

plt.matshow(last_dense_output, cmap='viridis')
plt.colorbar()
plt.title('Output of the Last Dense Layer')
plt.xlabel('Output Neurons')
plt.ylabel('Sample Index')
plt.show()

# Assuming 'attention_layer' is the name of your attention layer in the model
attention_model = Model(inputs=model.input, outputs=model.get_layer('attention_layer').output)
attention_weights = attention_model.predict(X_sample)  # Replace 'X_sample' with your data

# Visualize attention weights
plt.matshow(attention_weights, cmap='viridis')
plt.colorbar()
plt.title('Attention Weights')
plt.xlabel('Attention Output')
plt.ylabel('Sample Index')
plt.show()

# Assuming 'embedding_layer' is the name of your embedding layer
embedding_layer = model.get_layer('embedding_layer')
embeddings = embedding_layer.get_weights()[0]

# Using t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# Assuming 'model' has dropout layers for uncertainty estimation
predictions = []
for _ in range(100):  # Number of stochastic forward passes
    predictions.append(model.predict(X_sample, verbose=0))
predictions = np.array(predictions)

# Calculate mean and standard deviation
prediction_means = predictions.mean(axis=0)
prediction_stddevs = predictions.std(axis=0)

# Plot mean and standard deviation
plt.errorbar(range(len(prediction_means)), prediction_means, yerr=prediction_stddevs, fmt='o')
plt.title('Prediction with Uncertainty')
plt.xlabel('Sample Index')
plt.ylabel('Prediction')
plt.show()

for epoch, weights in enumerate(weight_change_cb.weight_history[1:]):
    plt.figure(figsize=(12, 6))
    for layer_idx, (initial, current) in enumerate(zip(initial_weights, weights)):
        weight_change = np.abs(current - initial)
        plt.subplot(1, len(initial_weights), layer_idx + 1)
        plt.hist(weight_change.flatten(), bins=50)
        plt.title(f'Layer {layer_idx+1} - Epoch {epoch+1}')
        plt.xlabel('Weight Change')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


for gradient, variable in zip(gradients, combined_model.trainable_variables):
    plt.hist(gradient.numpy().flatten(), bins=50)
    plt.title(f'Gradient Histogram for {variable.name}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.show()

conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

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
pltxlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

intermediate_model = Model(inputs=combined_model.input, outputs=[layer.output for layer in combined_model.layers if 'dense' in layer.name])
intermediate_outputs = intermediate_model.predict(X_val[:1])

for layer_output in intermediate_outputs:
    plt.hist(layer_output.flatten(), bins=50)
    plt.title('Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.show()

plt.figure(figsize=(6, 4))
plt.plot(lr_scheduler_cb.lr_rates, lr_scheduler_cb.losses)
plt.title('Loss vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xscale('log')  # Often plotted on a log scale for clarity
plt.show()

# Assuming you have a way to calculate or estimate feature importance
feature_importances = calculate_feature_importances(combined_model, X_val)

plt.bar(range(len(feature_importances)), feature_importances)
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

residuals = Y_val - y_pred  # Assuming Y_val is actual and y_pred is predicted
plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

plt.hist(y_pred, bins=50)
plt.title('Histogram of Predicted Values')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')
plt.show()

plt.scatter(Y_val, y_pred)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'k--', lw=4)
plt.show()

errors = np.abs(Y_val - y_pred)
plt.hist(errors, bins=100, cumulative=True, density=True, histtype='step', color='blue')
plt.title('Cumulative Distribution of Errors')
plt.xlabel('Error')
plt.ylabel('Cumulative Probability')
plt.show()

# Choose a layer to inspect
layer_to_inspect = 'dense_1'  # Replace with the name of the layer you want to inspect
activation_model = Model(inputs=combined_model.input, outputs=combined_model.get_layer(layer_to_inspect).output)
activations = activation_model.predict(X_val)

sns.heatmap(activations, cmap='viridis')
plt.title(f'Activation Heatmap for {layer_to_inspect}')
plt.xlabel('Neurons')
plt.ylabel('Samples')
plt.show()

# Assuming you choose a specific layer and feature for this visualization
selected_feature_index = 0  # Replace with the index of the feature you are interested in
selected_layer = 'dense_1'  # Replace with the layer of interest
activation_model = Model(inputs=combined_model.input, outputs=combined_model.get_layer(selected_layer).output)

# Create a scatter plot of feature values vs activation values
plt.scatter(X_val[:, selected_feature_index], activation_model.predict(X_val)[:, selected_feature_index])
plt.title(f'Input Feature vs Activation for {selected_layer}')
plt.xlabel(f'Feature {selected_feature_index} Value')
plt.ylabel(f'Activation of Neuron {selected_feature_index}')
plt.show()

tsne = TSNE(n_components=2, random_state=42)
selected_layer_output = activation_model.predict(X_val)  # Output of a selected layer
tsne_results = tsne.fit_transform(selected_layer_output)

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_true_classes)
plt.colorbar()
plt.title(f't-SNE of {selected_layer} Layer Outputs')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

individual_losses = tf.keras.losses.categorical_crossentropy(Y_val, y_pred)
plt.hist(individual_losses.numpy(), bins=50)
plt.title('Distribution of Losses')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.show()

# Assuming you have recorded weight changes in a callback
for layer_idx in range(len(initial_weights)):
    weight_updates = [np.sum(np.abs(weight_change_cb.weight_history[epoch][layer_idx] - initial_weights[layer_idx])) 
                      for epoch in range(len(weight_change_cb.weight_history))]
    plt.plot(weight_updates, label=f'Layer {layer_idx + 1}')

plt.title('Sum of Weight Updates Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Sum of Absolute Weight Updates')
plt.legend()
plt.show()

class_accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
plt.bar(range(len(class_accuracies)), class_accuracies)
plt.title('Class-wise Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.show()

# Assuming you have an activation layer named 'activation_1'
activation_output_model = Model(inputs=combined_model.input, outputs=combined_model.get_layer('activation_1').output)
activation_outputs = activation_output_model.predict(X_val)

plt.hist(activation_outputs.flatten(), bins=50)
plt.title('Activation Function Output Distribution')
plt.xlabel('Activation Output')
plt.ylabel('Frequency')
plt.show()

# Choose two features to analyze
feature1, feature2 = 0, 1  # Replace with the indices of the features you want to analyze
plt.scatter(X_val[:, feature1], X_val[:, feature2], c=y_pred)
plt.title('Feature Interaction')
plt.xlabel(f'Feature {feature1}')
plt.ylabel(f'Feature {feature2}')
plt.colorbar(label='Prediction')
plt.show()

errors = Y_val - y_pred  # Assuming a regression task
for i in range(X_val.shape[1]):  # For each feature
    plt.scatter(X_val[:, i], errors)
    plt.title(f'Error vs Feature {i}')
    plt.xlabel(f'Feature {i} Value')
    plt.ylabel('Error')
    plt.show()

activations = activation_model.predict(X_val)  # Activations from a selected layer
correlation_matrix = np.corrcoef(activations.T)

sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Activation Correlation Matrix')
plt.xlabel('Neurons')
plt.ylabel('Neurons')
plt.show()

improvements = np.diff(history['val_accuracy'])  # Assuming validation accuracy is a key metric
learning_rates = lr_scheduler_cb.lr_rates[1:]  # Exclude the initial learning rate

plt.scatter(learning_rates, improvements)
plt.title('Learning Rate Effectiveness')
plt.xlabel('Learning Rate')
plt.ylabel('Improvement in Validation Accuracy')
plt.show()

perturbed_X = np.copy(X_val)
perturbed_X[:, feature1] += 0.1  # Slight perturbation to a feature

perturbed_predictions = combined_model.predict(perturbed_X)
plt.scatter(X_val[:, feature1], perturbed_predictions - y_pred)
plt.title('Effect of Perturbation on Predictions')
plt.xlabel(f'Original Feature {feature1} Value')
plt.ylabel('Change in Prediction')
plt.show()

# Placeholder for a conceptual approach
# Calculating influence scores is complex and often requires a tailored approach
influence_scores = calculate_influence_scores(X_train, Y_train, X_val, combined_model)

# Visualizing the most influential samples
most_influential = np.argsort(influence_scores)[-10:]  # Top 10 influential samples
for idx in most_influential:
    plt.imshow(X_train[idx])
    plt.title(f'Most Influential Sample {idx}')
plt.axis('off')
plt.show()

# Assuming you have stored the activations of a specific layer after each epoch
for epoch in range(num_epochs):
    epoch_activations = stored_activations[epoch]  # Activations from a selected layer for a specific epoch
    plt.hist(epoch_activations.flatten(), bins=50, alpha=0.5, label=f'Epoch {epoch}')

plt.title('Layer Activations Over Epochs')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Choosing an intermediate layer for projection
intermediate_layer_model = Model(inputs=combined_model.input, outputs=combined_model.get_layer('dense_2').output)
projected_data = intermediate_layer_model.predict(X_val)

# Using PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(projected_data)

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_true_classes)
plt.title('Data Projection with Intermediate Layer')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.colorbar()
plt.show()

# Assuming a binary classification task
# Extracting the output of the last dense layer before the output layer
last_dense_layer_model = Model(inputs=combined_model.input, outputs=combined_model.get_layer('dense_3').output)
contributions = last_dense_layer_model.predict(X_val)

# Visualize the contribution of each neuron to the final decision
for neuron_index in range(contributions.shape[1]):
    plt.scatter(contributions[:, neuron_index], y_pred)
    plt.title(f'Neuron {neuron_index} Contribution to Predictions')
    plt.xlabel(f'Activation of Neuron {neuron_index}')
    plt.ylabel('Model Output')
    plt.show()

# This requires running multiple experiments with different hyperparameters
# For each set of hyperparameters, plot the model performance
hyperparameters = ['Hyperparam Set 1', 'Hyperparam Set 2', 'Hyperparam Set 3']
performances = [performance_set1, performance_set2, performance_set3]  # Replace with actual performance metrics

plt.bar(hyperparameters, performances)
plt.title('Hyperparameter Impact on Model Performance')
plt.xlabel('Hyperparameter Set')
plt.ylabel('Performance Metric')
plt.show()

# Assuming 'y_pred' are the output probabilities from the model
output_variance = np.var(y_pred, axis=0)
plt.bar(range(len(output_variance)), output_variance)
plt.title('Variance in Model Output')
plt.xlabel('Output Neuron Index')
plt.ylabel('Variance')
plt.show()

# Perform predictions with and without dropout
predictions_with_dropout = model.predict(X_val, training=True)  # Enable dropout
predictions_without_dropout = model.predict(X_val, training=False)  # Disable dropout

# Compare the distributions of predictions
plt.hist(predictions_with_dropout.flatten(), bins=50, alpha=0.5, label='With Dropout')
plt.hist(predictions_without_dropout.flatten(), bins=50, alpha=0.5, label='Without Dropout')
plt.title('Effect of Dropout on Model Predictions')
plt.xlabel('Prediction Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Extract activations with and without batch normalization
activations_without_bn = Model(inputs=model.input, outputs=model.get_layer('dense_1').output).predict(X_val)
activations_with_bn = Model(inputs=model.input, outputs=model.get_layer('batch_norm_layer').output).predict(X_val)

# Plot the activation distributions
plt.hist(activations_without_bn.flatten(), bins=50, alpha=0.5, label='Without Batch Normalization')
plt.hist(activations_with_bn.flatten(), bins=50, alpha=0.5, label='With Batch Normalization')
plt.title('Effect of Batch Normalization on Activations')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Extract activations with and without batch normalization
activations_without_bn = Model(inputs=model.input, outputs=model.get_layer('dense_1').output).predict(X_val)
activations_with_bn = Model(inputs=model.input, outputs=model.get_layer('batch_norm_layer').output).predict(X_val)

# Plot the activation distributions
plt.hist(activations_without_bn.flatten(), bins=50, alpha=0.5, label='Without Batch Normalization')
plt.hist(activations_with_bn.flatten(), bins=50, alpha=0.5, label='With Batch Normalization')
plt.title('Effect of Batch Normalization on Activations')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Slightly alter each feature and observe the change in predictions
for i in range(X_val.shape[1]):
    perturbed_X = np.copy(X_val)
    perturbed_X[:, i] += 0.1  # Small perturbation
    perturbed_predictions = model.predict(perturbed_X)
    sensitivity = np.mean(np.abs(perturbed_predictions - y_pred))
    plt.bar(i, sensitivity)

plt.title('Feature Sensitivity Analysis')
plt.xlabel('Feature Index')
plt.ylabel('Sensitivity')
plt.show()

# Placeholder for conceptual representation
# Visualizing loss landscape is complex and requires specific techniques like random direction search
loss_landscape = compute_loss_landscape(model, X_val, Y_val)
plt.imshow(loss_landscape, cmap='viridis', interpolation='bilinear')
plt.title('Loss Landscape')
plt.xlabel('Direction')
plt.ylabel('Loss')
plt.colorbar()
plt.show()

# Assuming you have performance metrics for different hyperparameter settings
for hyperparam, scores in hyperparam_performance_dict.items():
    plt.plot(hyperparam_values, scores, label=hyperparam)

plt.title('Hyperparameter Sensitivity')
plt.xlabel('Hyperparameter Value')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()



# Calculate the contribution of each feature to the error
feature_contributions = np.abs(X_val * (Y_val - y_pred))  # Assuming a simple linear relationship for illustration
mean_contributions = np.mean(feature_contributions, axis=0)
plt.bar(range(len(mean_contributions)), mean_contributions)
plt.title('Feature Contributions to Error')
plt.xlabel('Feature Index')
plt.ylabel('Mean Contribution to Error')
plt.show()


# Generate synthetic data (ensure it's in the same format as your training data)
synthetic_data = generate_synthetic_data()
synthetic_predictions = model.predict(synthetic_data)

plt.hist(synthetic_predictions, bins=50)
plt.title('Model Response to Synthetic Data')
plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.show()

# Assuming a ReLU activation function for illustration
relu_activations = Model(inputs=model.input, outputs=model.get_layer('relu_layer').output).predict(X_val)
saturated_neurons = np.mean(relu_activations == 0, axis=0)

plt.bar(range(len(saturated_neurons)), saturated_neurons)
plt.title('ReLU Neuron Saturation')
plt.xlabel('Neuron Index')
plt.ylabel('Proportion of Saturated Neurons')
plt.show()

# Assuming you have models with different weight initializations
initial_accuracies = [initial_accuracy_model1, initial_accuracy_model2, ...]

plt.bar(['Initialization 1', 'Initialization 2', ...], initial_accuracies)
plt.title('Impact of Weight Initialization on Initial Accuracy')
plt.xlabel('Weight Initialization Strategy')
plt.ylabel('Initial Accuracy')
plt.show()

# Placeholder for conceptual representation
# Requires tracking parameter values across training iterations
parameter_values = get_optimizer_trajectory(model)
plt.plot(parameter_values)
plt.title('Optimization Trajectory in Parameter Space')
plt.xlabel('Training Iteration')
plt.ylabel('Parameter Value')
plt.show()

# Analyze the distribution of each feature in the input data
for i in range(X_val.shape[1]):
    plt.hist(X_val[:, i], bins=50)
    plt.title(f'Distribution of Feature {i}')
    plt.xlabel(f'Feature {i} Value')
    plt.ylabel('Frequency')
    plt.show()

# Extract outputs from an early layer
early_layer_output = Model(inputs=model.input, outputs=model.get_layer('early_layer').output).predict(X_val)

# Plotting scatter plots for each pair of neurons
for i in range(early_layer_output.shape[1]):
    for j in range(i + 1, early_layer_output.shape[1]):
        plt.scatter(early_layer_output[:, i], early_layer_output[:, j])
        plt.title(f'Layer Output Scatter Plot: Neurons {i} and {j}')
        plt.xlabel(f'Neuron {i} Output')
        plt.ylabel(f'Neuron {j} Output')
        plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'])
plt.title('Training Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

# Assuming 'relu_activations' and 'sigmoid_activations' are activations from layers with ReLU and Sigmoid respectively
plt.hist(relu_activations.flatten(), bins=50, alpha=0.5, label='ReLU')
plt.hist(sigmoid_activations.flatten(), bins=50
# Assuming you have models trained with and without regularization
plt.plot(history_no_reg['val_loss'], label='Without Regularization')
plt.plot(history_with_reg['val_loss'], label='With Regularization')
plt.title('Effect of Regularization on Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

misclassified_indices = np.where(np.argmax(y_pred, axis=1) != np.argmax(Y_val, axis=1))[0]
for i in misclassified_indices[:10]:  # Display first 10 misclassified examples
    plt.imshow(X_val[i].reshape(image_shape))  # Reshape if necessary
    plt.title(f'True: {np.argmax(Y_val[i])}, Pred: {np.argmax(y_pred[i])}')
    plt.show()

for layer in model.layers:
    if hasattr(layer, 'get_weights'):
        weights, biases = layer.get_weights()
        plt.hist(weights.flatten(), bins=50, alpha=0.5, label='Weights')
        plt.hist(biases.flatten(), bins=50, alpha=0.5, label='Biases')
        plt.title(f'Parameter Histogram of {layer.name}')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

predicted_classes = np.argmax(y_pred, axis=1)
plt.hist(predicted_classes, bins=np.arange(0, num_classes + 1) - 0.5, rwidth=0.8)
plt.xticks(range(num_classes))
plt.title('Distribution of Predicted Classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Placeholder for conceptual representation
# Requires sampling the loss function over a grid of parameter values
param1_range = np.linspace(start=-1, stop=1, num=50)
param2_range = np.linspace(start=-1, stop=1, num=50)
loss_surface = compute_loss_surface(model, param1_range, param2_range, X_val, Y_val)

plt.contourf(param1_range, param2_range, loss_surface, levels=50, cmap='viridis')
plt.title('Loss Function Surface')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.colorbar()
plt.show()

# Example with a Sigmoid activation function
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
sigmoid_derivative = sigmoid * (1 - sigmoid)

plt.plot(z, sigmoid_derivative)
plt.title('Derivative of the Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Derivative')
plt.show()

false_positives = X_val[(y_true_classes == 0) & (y_pred_classes == 1)]
false_negatives = X_val[(y_true_classes == 1) & (y_pred_classes == 0)]

# Analyzing the distributions or characteristics of false positives and negatives
# For instance, plotting the mean or median of these samples
plt.plot(np.mean(false_positives, axis=0), label='False Positives')
plt.plot(np.mean(false_negatives, axis=0), label='False Negatives')
plt.title('Comparison of False Positives and False Negatives')
plt.xlabel('Feature Index')
plt.ylabel('Average Feature Value')
plt.legend()
plt.show()

# Assuming 'neuron_outputs_over_time' is a list of neuron outputs recorded at different epochs
for neuron_index in range(number_of_neurons):
    plt.plot([output[neuron_index] for output in neuron_outputs_over_time])
    plt.title(f'Output of Neuron {neuron_index} Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Neuron Output')
    plt.show()

# Example using outputs from an intermediate layer
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(intermediate_layer_output)

plt.scatter(intermediate_layer_output[:, 0], intermediate_layer_output[:, 1], c=clusters)
plt.title('Cluster Analysis of Intermediate Layer Outputs')
plt.xlabel('Output Dimension 1')
plt.ylabel('Output Dimension 2')
plt.colorbar()
plt.show()

# Assuming you have a function that prunes neurons and returns model performance
neuron_counts = []
model_performances = []
for n in range(max_neuron_count, min_neuron_count, step):
    neuron_counts.append(n)
    performance = prune_neurons_and_evaluate(model, n)  # Custom function to prune neurons
    model_performances.append(performance)

plt.plot(neuron_counts, model_performances)
plt.title('Neuron Pruning Analysis')
plt.xlabel('Number of Neurons')
plt.ylabel('Model Performance')
plt.show()
# Assuming you have a function that prunes neurons and returns model performance
neuron_counts = []
model_performances = []
for n in range(max_neuron_count, min_neuron_count, step):
    neuron_counts.append(n)
    performance = prune_neurons_and_evaluate(model, n)  # Custom function to prune neurons
    model_performances.append(performance)

plt.plot(neuron_counts, model_performances)
plt.title('Neuron Pruning Analysis')
plt.xlabel('Number of Neurons')
plt.ylabel('Model Performance')
plt.show()

# Assuming 'gradient_norms' is a list of gradient norms for each layer after a training step
for layer_index, norms in enumerate(gradient_norms):
    plt.plot(norms, label=f'Layer {layer_index}')

plt.title('Gradient Flow Through Layers')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.legend()
plt.show()

original_performance = model.evaluate(X_val, Y_val)
shuffled_performances = []
for i in range(num_features):
    X_val_shuffled = np.copy(X_val)
    np.random.shuffle(X_val_shuffled[:, i])  # Shuffle a single feature
    performance = model.evaluate(X_val_shuffled, Y_val)
    shuffled_performances.append(performance)

plt.plot(shuffled_performances)
plt.axhline(y=original_performance, color='r', linestyle='--', label='Original Performance')
plt.title('Impact of Input Shuffling on Performance')
plt.xlabel('Feature Index')
plt.ylabel('Performance')
plt.legend()
plt.show()

# Assuming 'weight_history' records the weights of a specific layer at each epoch
epochs = range(len(weight_history))
for i in range(num_neurons_in_layer):
    neuron_weight_changes = [np.linalg.norm(weight_history[epoch][i] - weight_history[0][i]) for epoch in epochs]
    plt.plot(epochs, neuron_weight_changes, label=f'Neuron {i}')

plt.title('Neuron Weight Convergence Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Weight Change (from initial)')
plt.legend()
plt.show()

# Assuming 'prediction_intervals' is a function that calculates prediction intervals
lower_bounds, upper_bounds = prediction_intervals(model, X_val, confidence=0.95)
plt.fill_between(range(len(Y_val)), lower_bounds, upper_bounds, color='gray', alpha=0.5)
plt.plot(Y_val, label='Actual Values')
plt.plot(y_pred,label='Predicted Values', color='red')
plt.title('Prediction Interval Analysis')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

# Assuming 'attention_layer_output' is the output of an attention layer
attention_layer_output = Model(inputs=model.input, outputs=model.get_layer('attention_layer').output).predict(X_val)

for i in range(num_samples_to_visualize):
    plt.matshow(attention_layer_output[i], cmap='viridis')
    plt.title(f'Attention Map for Sample {i}')
    plt.xlabel('Attention Weights')
    plt.ylabel('Features')
    plt.colorbar()
    plt.show()

# Assuming 'predict_with_uncertainty' is a function that generates predictions with uncertainty
predictions, uncertainties = predict_with_uncertainty(model, X_val)
plt.errorbar(range(len(predictions)), predictions, yerr=uncertainties, fmt='o')
plt.title('Neural Network Predictions with Uncertainty')
plt.xlabel('Sample Index')
plt.ylabel('Prediction')
plt.show()

# Assuming 'calculate_feature_importance' is a function to calculate feature importance
for epoch in range(num_epochs):
    feature_importances = calculate_feature_importance(model, X_val, epoch)
    plt.plot(feature_importances, label=f'Epoch {epoch}')

plt.title('Feature Importance Over Training Epochs')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.legend()
plt.show()


# Assuming 'embedding_layer_name' is the name of your embedding layer
embedding_layer = Model(inputs=model.input, outputs=model.get_layer(embedding_layer_name).output)
embeddings = embedding_layer.predict(X_val_categorical_part)  # Categorical part of X_val

# Using t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=y_val)
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()
plt.show()

residuals = Y_val - y_pred  # Assuming Y_val is the actual value and y_pred is the predicted value
plt.scatter(y_pred, residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Assuming 'layer_outputs' is a list of outputs from different layers for a given input
for i, layer_output in enumerate(layer_outputs):
    sensitivities = np.std(layer_output, axis=0)
    plt.plot(sensitivities, label=f'Layer {i+1}')

plt.title('Layer-wise Feature Sensitivity')
plt.xlabel('Feature Index')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

outlier_indices = detect_outliers(X_val)  # Assuming a function to detect outliers
outlier_predictions = model.predict(X_val[outlier_indices])

plt.hist(outlier_predictions, bins=50)
plt.title('Model\'s Reaction to Outliers')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')
plt.show()

# Assuming 'segment_data' is a function that segments the data
segments = segment_data(X_val)
segment_performances = {}
for segment_name, segment_data in segments.items():
    performance = model.evaluate(segment_data, Y_val)
    segment_performances[segment_name] = performance

plt.bar(segment_performances.keys(), segment_performances.values())
plt.title('Model Performance on Different Data Segments')
plt.xlabel('Data Segment')
plt.ylabel('Performance')
plt.xticks(rotation=45)
plt.show()# Assuming 'segment_data' is a function that segments the data
segments = segment_data(X_val)
segment_performances = {}
for segment_name, segment_data in segments.items():
    performance = model.evaluate(segment_data, Y_val)
    segment_performances[segment_name] = performance

plt.bar(segment_performances.keys(), segment_performances.values())
plt.title('Model Performance on Different Data Segments')
plt.xlabel('Data Segment')
plt.ylabel('Performance')
plt.xticks(rotation=45)
plt.show()

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()[0]  # Get the weights
        plt.hist(weights.flatten(), bins=50, alpha=0.5, label=f'{layer.name} (Dense)')
    elif isinstance(layer, tf.keras.layers.Dropout):
        # For Dropout layers, you might want to analyze the rate or effect
        plt.axvline(x=layer.rate, label=f'{layer.name} (Dropout)', linestyle='--')
    # Add more layer types as needed

plt.title('Weight Distribution by Layer Type')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()



                                                                            



















