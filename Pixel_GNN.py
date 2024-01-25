import os
import numpy as np
import tensorflow as tf
from models import call_model
from inference_tools.graphs import get_input_target_graphs
from inference_tools.outputs import graphsTuple_to_nx
from models.edgeClassifier_directed import utils_tf_v2 as utils_tf
from models.edgeClassifier_directed import graphs
from inference_tools import load_yaml

# Configure TensorFlow and GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load configuration
config_file = 'path/to/config.yaml'
config = load_yaml(config_file)

# Extract data and checkpoint directories from the config
data_directory = config['input']['data_directory']
ckpt_directory = config['input']['ckpt_directory']

# Load the CMS tracking data as graph structures
inputs_G, targets_G, event_ids = get_input_target_graphs(data_directory)

# Initialize the GNN model based on the provided configuration
model_name = config['GNN_parameters']['model_name']
model = call_model(model_name, **config['GNN_parameters']['model_hyperparameters'])

# Set up checkpointing to load the model weights
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(ckpt_directory)).expect_partial()

# TensorFlow function for making predictions with the GNN
@tf.function
def prediction(inputs):
    return model(inputs, concat=config['GNN_parameters']['concat'])
for i, input_graph in enumerate(inputs_G):
    input_G = utils_tf.data_dicts_to_graphs_tuple(input_graph)
    predicted_graph = prediction(input_G)

    # Convert predicted graphs to networkx format for analysis
    nx_graph = graphsTuple_to_nx(predicted_graph, input_graph, targets_G[i],
                                 reduced_graph=config['output']['reduced_graph'],
                                 threshold=config['output']['threshold'],
                                 save=config['output']['save'],
                                 path_to_saveGraph=config['output']['path_to_saveGraph'],
                                 evt_id=event_ids[i])
    # Further analysis and processing of nx_graph
    # ...
def preprocess_data(graphs):
    # Implement preprocessing steps like normalization, feature engineering, etc.
    # ...
    return processed_graphs

inputs_G = preprocess_data(inputs_G)

optimizer = tf.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(input_graph, target_graph):
    with tf.GradientTape() as tape:
        predictions = model(input_graph, training=True)
        loss = compute_loss(predictions, target_graph)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(num_epochs):
    for input_graph, target_graph in zip(inputs_G, targets_G):
        loss = train_step(input_graph, target_graph)
        # Log loss, validate model, etc.
def compute_loss(predictions, targets):
    return tf.nn.softmax_cross_entropy_with_logits(logits=predictions.edges, labels=targets.edges)
def evaluate_model(inputs, targets):
    predictions = model(inputs, training=False)
    # Calculate evaluation metrics
    accuracy = compute_accuracy(predictions, targets)
    precision = compute_precision(predictions, targets)
    recall = compute_recall(predictions, targets)
    return accuracy, precision, recall

# Example usage during or after the training loop
for input_graph, target_graph in zip(inputs_G, targets_G):
    accuracy, precision, recall = evaluate_model(input_graph, target_graph)
    # Log or print the metrics
# Save the model
model.save('path/to/save/model')

# Load the model
loaded_model = tf.keras.models.load_model('path/to/save/model')
def process_predictions(predictions):
    # Implement any post-prediction processing
    # This could involve transforming the data into a desired format,
    # filtering results, combining with other data sources, etc.
    # ...
    return processed_data

processed_data = process_predictions(predicted_graph)
from tensorboard.plugins.hparams import api as hp

# Example: Logging hyperparameters and metrics
with tf.summary.create_file_writer('path/to/logs').as_default():
    hp.hparams({'learning_rate': 0.001, 'num_epochs': 10})
    tf.summary.scalar('accuracy', accuracy, step=epoch)
    tf.summary.scalar('precision', precision, step=epoch)
    tf.summary.scalar('recall', recall, step=epoch)
