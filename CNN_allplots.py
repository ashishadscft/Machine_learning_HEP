import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback for recording loss and accuracy each epoch
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_accuracy'))

training_callback = TrainingCallback()

# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2, callbacks=[training_callback])

# Loss and Accuracy Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(training_callback.losses, label='Training Loss')
plt.plot(training_callback.val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_callback.accuracy, label='Training Accuracy')
plt.plot(training_callback.val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# Visualizing Feature Maps
test_image = test_images[0].reshape(1, 28, 28, 1)
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_image)

# Plot the feature maps
for layer_activations in activations:
    n_features = layer_activations.shape[-1]
    size = layer_activations.shape[1]
    n_cols = n_features // 16
    display_grid = np.zeros((size * n_cols, 16 * size))

    for col in range(n_cols):
        for row in range(16):
            channel_image = layer_activations[0, :, :, col * 16 + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(model.layers[0].name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')




from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the values from the test dataset
test_pred = model.predict(test_images)
test_pred_classes = np.argmax(test_pred, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


from sklearn.metrics import roc_curve, auc

# Compute ROC curve for a specific class
fpr, tpr, thresholds = roc_curve(test_labels, test_pred[:, 0], pos_label=0)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic - class 0')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import precision_recall_curve

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(test_labels, test_pred[:, 0], pos_label=0)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.title('Precision-Recall curve - class 0')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# Extracting model layers
layer_names = [layer.name for layer in model.layers]

# Applying the model to an input
input_image = train_images[0].reshape((1, 28, 28, 1))
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(input_image)

# Plotting activations of each layer
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    plt.figure(figsize=(n_features * 0.5, 2))
    plt.title(layer_name)

    for i in range(n_features):
        plt.subplot(2, n_features, i+1)
        plt.imshow(layer_activation[0, :, :, i], aspect='auto', cmap='viridis')
        plt.axis('off')

    plt.show()


class GradientCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.gradients = []

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            inputs = self.model.input
            outputs = self.model.output
            tape.watch(inputs)
            preds = self.model(inputs)
            loss = self.model.compiled_loss(train_labels, preds)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.gradients.append(grads)

gradient_callback = GradientCallback()

# Add the gradient callback to the training process
model.fit(train_images, train_labels, epochs=5, callbacks=[gradient_callback])

# Plotting the gradient flow
plt.figure(figsize=(10, 8))
for epoch, grads in enumerate(gradient_callback.gradients):
    plt.plot(np.mean([np.mean(g) for g in grads]), label=f'Epoch {epoch + 1}')
plt.title('Gradient Flow Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Gradient')
plt.legend()
plt.show()


class WeightDistributionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        layer_weights = self.model.layers[0].get_weights()[0]  # For the first layer
        plt.figure(figsize=(6, 4))
        plt.hist(layer_weights.flatten(), bins=50)
        plt.title(f'Weight Distribution in Layer 1 - Epoch {epoch + 1}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()

weight_callback = WeightDistributionCallback()

# Train the model with the weight distribution callback
model.fit(train_images, train_labels, epochs=5, callbacks=[weight_callback])


class WeightChangeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.initial_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        final_weights = self.model.get_weights()
        weight_changes = [np.mean(abs(iw - fw)) for iw, fw in zip(self.initial_weights, final_weights)]
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(weight_changes)), weight_changes)
        plt.title('Weight Changes from Initial to Final Epoch')
        plt.xlabel('Layer')
        plt.ylabel('Average Weight Change')
        plt.show()

weight_change_callback = WeightChangeCallback()

# Train the model with the weight change callback
model.fit(train_images, train_labels, epochs=5, callbacks=[weight_change_callback])


class ActivationHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index

    def on_epoch_end(self, epoch, logs=None):
        layer_output = self.model.layers[self.layer_index].output
        intermediate_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_output)
        intermediate_prediction = intermediate_model.predict(train_images[:1000])  # Sample some data

        plt.figure(figsize=(6, 4))
        plt.hist(intermediate_prediction.flatten(), bins=100)
        plt.title(f'Activation Histogram for Layer {self.layer_index} - Epoch {epoch + 1}')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.show()

activation_histogram_callback = ActivationHistogramCallback(layer_index=2)

# Train the model with the activation histogram callback
model.fit(train_images, train_labels, epochs=5, callbacks=[activation_histogram_callback])


def plot_layer_output_heatmap(model, layer_index, input_image):
    layer_output = model.layers[layer_index].output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    intermediate_prediction = intermediate_model.predict(input_image.reshape((1, 28, 28, 1)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(intermediate_prediction[0, :, :, :], cmap='viridis')
    plt.title(f'Output Heatmap for Layer {layer_index}')
    plt.show()

plot_layer_output_heatmap(model, layer_index=2, input_image=train_images[0])

def plot_neuron_lifetime_sparsity(model, layer_index, dataset):
    layer_output = model.layers[layer_index].output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    activations = intermediate_model.predict(dataset)

    neuron_activations = np.mean(activations > 0, axis=0).flatten()  # Percentage of time neuron is active
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(neuron_activations)), neuron_activations)
    plt.title(f'Neuron Lifetime Sparsity in Layer {layer_index}')
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Frequency')
    plt.show()

plot_neuron_lifetime_sparsity(model, layer_index=2, dataset=train_images)

from sklearn.manifold import TSNE

def plot_tsne_of_layer_output(model, layer_index, dataset, labels):
    layer_output = model.layers[layer_index].output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    intermediate_prediction = intermediate_model.predict(dataset)

    # Flattening the output and using only a subset of data for t-SNE for computational efficiency
    flattened_output = intermediate_prediction.reshape(intermediate_prediction.shape[0], -1)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(flattened_output[:1000])  # Using a subset for efficiency

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels[:1000], palette=sns.color_palette("hsv", 10), legend='full')
    plt.title(f't-SNE of Layer {layer_index} Output')
    plt.show()

plot_tsne_of_layer_output(model, layer_index=2, dataset=train_images, labels=train_labels)


def plot_inter_layer_correlation(model, dataset):
    layer_outputs = [model.layers[i].output for i in range(len(model.layers))]
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    intermediate_prediction = intermediate_model.predict(dataset[:1000])  # Use a subset for efficiency

    correlations = []
    for i in range(len(layer_outputs)):
        flattened_output_i = intermediate_prediction[i].reshape(intermediate_prediction[i].shape[0], -1)
        for j in range(i+1, len(layer_outputs)):
            flattened_output_j = intermediate_prediction[j].reshape(intermediate_prediction[j].shape[0], -1)
            corr = np.corrcoef(flattened_output_i.T, flattened_output_j.T)[0:len(flattened_output_i.T), len(flattened_output_i.T):]
            correlations.append((i, j, np.mean(corr)))
    plt.figure(figsize=(12, 8))
    for corr in correlations:
        plt.scatter(corr[0], corr[1], s=corr[2]*100, c='blue')
    plt.title('Inter-Layer Correlation Matrix')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.colorbar()
    plt.show()

plot_inter_layer_correlation(model, train_images)

def plot_node_influence(model, layer_index, dataset, labels):
    original_accuracy = model.evaluate(dataset, labels, verbose=0)[1]
    layer_weights = model.layers[layer_index].get_weights()
    influences = []

    for i in range(layer_weights[0].shape[1]):
        new_weights = [np.copy(w) for w in layer_weights]
        new_weights[0][:, i] = 0  # Zeroing out the weights of the i-th node
        model.layers[layer_index].set_weights(new_weights)
        new_accuracy = model.evaluate(dataset, labels, verbose=0)[1]
        influences.append(original_accuracy - new_accuracy)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(influences)), influences)
    plt.title(f'Node Influence in Layer {layer_index}')
    plt.xlabel('Node Index')
    plt.ylabel('Influence on Accuracy')
    plt.show()

    model.layers[layer_index].set_weights(layer_weights)  # Reset to original weights

plot_node_influence(model, layer_index=2, dataset=test_images, labels=test_labels)


from tensorflow_model_optimization.sparsity import keras as sparsity

def prune_model(model, dataset, labels):
    epochs = 2
    end_step = np.ceil(len(dataset) / 32).astype(np.int32) * epochs
    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=0, end_step=end_step)}

    pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [sparsity.UpdatePruningStep()]
    pruned_model.fit(dataset, labels, epochs=epochs, callbacks=callbacks)

    return pruned_model
pruned_model = prune_model(model, train_images, train_labels)

# Compare the performance of the original and pruned models
original_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
pruned_accuracy = pruned_model.evaluate(test_images, test_labels, verbose=0)[1]

plt.figure(figsize=(6, 4))
plt.bar(['Original', 'Pruned'], [original_accuracy, pruned_accuracy])
plt.title('Model Accuracy: Original vs Pruned')
plt.ylabel('Accuracy')
plt.show()

def plot_activation_over_position(model, layer_index, input_image):
    layer_output = model.layers[layer_index].output
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    activation = intermediate_model.predict(input_image.reshape((1, 28, 28, 1)))

    plt.figure(figsize=(15, 5))
    for i in range(activation.shape[3]):  # Iterate over the channels
        plt.plot(activation[0, :, :, i].flatten(), label=f'Channel {i}')
    plt.title(f'Activation Over Position for Layer {layer_index}')
    plt.xlabel('Position')
    plt.ylabel('Activation')
    plt.legend()
    plt.show()

plot_activation_over_position(model, layer_index=2, input_image=train_images[0])


from sklearn.decomposition import PCA

def plot_embedding_layer(model, layer_index):
    if 'embedding' in model.layers[layer_index].name:
        embeddings = model.layers[layer_index].get_weights()[0]
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        plt.title(f'PCA of Embeddings from Layer {layer_index}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
    else:
        print("Selected layer is not an embedding layer.")

def plot_output_sensitivity(model, input_image, epsilon=0.01):
    original_output = model.predict(input_image.reshape((1, 28, 28, 1)))
    sensitivities = []

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            perturbed_image = np.copy(input_image)
            perturbed_image[i, j] += epsilon
            perturbed_output = model.predict(perturbed_image.reshape((1, 28, 28, 1)))
            sensitivity = np.sum(np.abs(original_output - perturbed_output))
            sensitivities.append(sensitivity)

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(sensitivities).reshape(input_image.shape), cmap='viridis')
    plt.title('Output Sensitivity to Input Perturbations')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

plot_output_sensitivity(model, train_images[0])


# Requires a specific architecture, usually a CNN with global average pooling before the final layer
def plot_cam(model, img, class_idx, last_conv_layer_name):
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    cam_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Then create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Get the gradient of the class output w.r.t. the feature map
    with tf.GradientTape() as tape:
        last_conv_layer_output = cam_model(img)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_class_channel = preds[:, class_idx]

    # Compute the gradient of the class output w.r.t. the feature map
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its corresponding gradient's importance
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Generate the heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Display the heatmap
    plt.matshow(heatmap)
    plt.show()


def plot_activation_correlation_matrix(model, layer_index, dataset):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_index].output)
    activations = intermediate_layer_model.predict(dataset)
    flattened_activations = activations.reshape(activations.shape[0], -1)

    correlation_matrix = np.corrcoef(flattened_activations.T)

    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, cmap='viridis')
    plt.title(f'Activation Correlation Matrix for Layer {layer_index}')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')
    plt.show()

plot_activation_correlation_matrix(model, layer_index=2, dataset=train_images[:1000])  # Using a subset of data for efficiency


def plot_layer_output_variance(model, dataset):
    variances = []
    for layer in model.layers:
        intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
        activations = intermediate_layer_model.predict(dataset)
        variances.append(np.var(activations))

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.layers)), variances)
    plt.title('Layer Output Variance')
    plt.xlabel('Layer Index')
    plt.ylabel('Variance')
    plt.show()

plot_layer_output_variance(model, train_images[:1000])  # Using a subset of data for efficiency


def plot_feature_map_overlap(model, layer_index, input_image):
    if 'conv' in model.layers[layer_index].name:
        # Getting the output of the layer
        layer_output = model.layers[layer_index].output
        intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
        feature_maps = intermediate_model.predict(input_image.reshape((1, 28, 28, 1)))

        # Calculating overlap
        num_feature_maps = feature_maps.shape[-1]
        overlap_matrix = np.zeros((num_feature_maps, num_feature_maps))

        for i in range(num_feature_maps):
            for j in range(num_feature_maps):
                overlap = np.sum((feature_maps[0, :, :, i] > 0) & (feature_maps[0, :, :, j] > 0))
                overlap_matrix[i, j] = overlap
                
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, cmap='viridis')
        plt.title(f'Feature Map Overlap in Layer {layer_index}')
        plt.xlabel('Feature Map')
        plt.ylabel('Feature Map')
        plt.show()
    else:
        print("Selected layer is not a convolutional layer.")


class FilterEvolutionCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.filter_weights = []

    def on_epoch_end(self, epoch, logs=None):
        filters = self.model.layers[self.layer_index].get_weights()[0]
        self.filter_weights.append(filters)

filter_evolution_callback = FilterEvolutionCallback(layer_index=1)
model.fit(train_images, train_labels, epochs=5, callbacks=[filter_evolution_callback])

# Plotting filter evolution
def plot_filter_evolution(filter_weights):
    num_epochs = len(filter_weights)
    num_filters = filter_weights[0].shape[-1]

    for i in range(num_filters):
        plt.figure(figsize=(num_epochs * 2, 2))
        for epoch in range(num_epochs):
            filter = filter_weights[epoch][:, :, :, i]
            plt.subplot(1, num_epochs, epoch + 1)
            plt.imshow(filter[:, :, 0], cmap='viridis')
            plt.axis('off')
        plt.show()

plot_filter_evolution(filter_evolution_callback.filter_weights)



