import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

class TrackGNN(torch.nn.Module):
    def __init__(self, input_features):
        super(TrackGNN, self).__init__()
        self.conv1 = GCNConv(input_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 3)  # Predicting 3D momentum vector

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Mean pooling over each graph in the batch
        x = self.fc(x)
        return x

# Loading dataset
dataset = ...  
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrackGNN(input_features=dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

num_epochs = 50
train_losses, test_losses, train_maes, test_maes, train_r2s, test_r2s = [], [], [], [], [], []

for epoch in range(num_epochs):
    train_loss = train()
    test_loss = test(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Calculate and record MAE
    train_mae = calculate_mae(train_loader)
    test_mae = calculate_mae(test_loader)
    train_maes.append(train_mae)
    test_maes.append(test_mae)

    # Calculate and record R2 Score
    train_r2 = calculate_r2(train_loader)
    test_r2 = calculate_r2(test_loader)
    train_r2s.append(train_r2)
    test_r2s.append(test_r2)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()

def calculate_mae(loader):
    model.eval()
    actuals, predictions = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            actuals.extend(data.y.cpu().numpy())
            predictions.extend(out.cpu().numpy())
    return mean_absolute_error(actuals, predictions)

def calculate_r2(loader):
    model.eval()
    actuals, predictions = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            actuals.extend(data.y.cpu().numpy())
            predictions.extend(out.cpu().numpy())
    return r2_score(actuals, predictions)

# Plot Mean Absolute Error Over Epochs
plt.figure(figsize=(8, 5))
plt.plot(train_maes, label='Train MAE')
plt.plot(test_maes, label='Test MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error Over Epochs')
plt.legend()
plt.show()

# Plot R2 Score Over Epochs
plt.figure(figsize=(8, 5))
plt.plot(train_r2s, label='Train R2 Score')
plt.plot(test_r2s, label='Test R2 Score')
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.title('R2 Score Over Epochs')
plt.legend()
plt.show()

# Visualizing Predictions vs Actual Values
model.eval()
actuals, predictions = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        actuals.extend(data.y.cpu().numpy())
        predictions.extend(out.cpu().numpy())

plt.figure(figsize=(8, 8))
plt.scatter(actuals, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--', lw=4)
plt.show()

# Histogram of Prediction Errors
errors = [pred - act for pred, act in zip(predictions, actuals)]
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()

# Assuming you have a scheduler and you're updating it in your training loop
# Also assuming you have a list 'learning_rates' that you update each epoch after scheduler.step()

plt.figure(figsize=(8, 5))
plt.plot(learning_rates)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule Over Epochs')
plt.show()

# Assuming 'feature_importances' is a list/array of importances
plt.figure(figsize=(8, 5))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Assuming 'embeddings' is a numpy array of embeddings from your model
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Model Embeddings')
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 8))
plt.scatter(pca_results[:, 0], pca_results[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Model Embeddings')
plt.show()


import networkx as nx

# Assuming 'data' is a PyTorch Geometric data object representing a graph
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title('Graph Structure')
plt.show()

# Visualize Layer Outputs and Activations
def plot_activation_map(layer_output, title="Activation Map"):
    activation_map = layer_output.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    sns.heatmap(activation_map, annot=False, cmap='viridis')
    plt.title(title)
    plt.show()

# Call this function with the output of the desired layer


from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Assuming binary classification and 'y_true' and 'y_scores' are true and predicted scores
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Assuming 'attention_weights' is an array/list of attention weights from your model
plt.figure(figsize=(8, 5))
plt.hist(attention_weights, bins=30)
plt.xlabel('Attention Weights')
plt.ylabel('Frequency')
plt.title('Distribution of Attention Weights')
plt.show()

# Visualizing node embeddings in a scatter plot
# Assuming 'node_embeddings' is a tensor of node embeddings from your GNN
plt.figure(figsize=(8, 5))
plt.hist(node_embeddings.detach().cpu().numpy().flatten(), bins=30)
plt.xlabel('Node Embedding Values')
plt.ylabel('Frequency')
plt.title('Histogram of Node Embeddings')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming 'y_true' and 'y_pred' are your true and predicted labels
conf_mat = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)

# Call this function after the loss.backward() call in your training loop
plot_grad_flow(model.named_parameters())

# Learning rate analysis
# Assuming you have different learning rates for different layers
layer_lr = {'layer1': [], 'layer2': [], ...}  # Initialize for each layer

for epoch in range(num_epochs):
    # Update this after each optimizer.step()
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer = name.split('.')[0]  # Adjust based on your model's parameter naming
            layer_lr[layer].append(optimizer.param_groups[0]['lr'])

# Now plot
plt.figure(figsize=(10, 5))
for layer, lrs in layer_lr.items():
    plt.plot(lrs, label=layer)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Layer-wise Learning Rate During Training')
plt.legend()
plt.show()

import networkx as nx

# Visualizing the graph structure
# Assuming 'data' is a PyTorch Geometric data object representing a graph
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title('Graph Structure')
plt.show()


# Assuming 'node_features' is a numpy array of node features
plt.figure(figsize=(10, 6))
sns.heatmap(node_features, annot=False, cmap='viridis')
plt.xlabel('Feature Index')
plt.ylabel('Node Index')
plt.title('Heatmap of Node Features')
plt.show()

# Edge feature analysis
# Assuming 'edge_weights' is a tensor of edge weights from your model
plt.figure(figsize=(8, 5))
plt.hist(edge_weights.detach().cpu().numpy(), bins=30)
plt.xlabel('Edge Weight')
plt.ylabel('Frequency')
plt.title('Edge Weight Distribution')
plt.show()

# Assuming 'uncertainties' is an array of uncertainty values associated with predictions
plt.figure(figsize=(8, 5))
plt.hist(uncertainties, bins=30)
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.title('Model Uncertainty Distribution')
plt.show()

# Assuming 'losses' is a list of loss values, one for each graph in the batch
plt.figure(figsize=(8, 5))
plt.hist(losses, bins=30)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Loss Distribution Across Graphs')
plt.show()

# Assuming 'feature_importances' is an array of feature importance scores
plt.figure(figsize=(8, 5))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.title('Node Feature Importance')
plt.show()

# Assuming 'path_lengths' is a list of path lengths in your graph
plt.figure(figsize=(8, 5))
plt.hist(path_lengths, bins=30)
plt.xlabel('Path Length')
plt.ylabel('Frequency')
plt.title('Path Length Distribution in Graphs')
plt.show()

# Assuming 'data_list' is your list of graph data objects and 'losses' is a corresponding list of losses
high_loss_graphs = [data_list[i] for i in np.argsort(losses)[-3:]]  # Top 3 high-loss graphs
low_loss_graphs = [data_list[i] for i in np.argsort(losses)[:3]]   # Top 3 low-loss graphs

for graph in high_loss_graphs + low_loss_graphs:
    G = nx.Graph()
    G.add_edges_from(graph.edge_index.t().tolist())
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.show()

# Calculating and visualizing network metrics like clustering coefficient, centrality, etc.
import networkx as nx

# Assuming 'G' is a NetworkX graph
clustering_coeffs = nx.clustering(G).values()
plt.figure(figsize=(8, 5))
plt.hist(clustering_coeffs, bins=30)
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.title('Clustering Coefficient Distribution')
plt.show()

centralities = nx.eigenvector_centrality_numpy(G)
plt.figure(figsize=(8, 5))
plt.hist(list(centralities.values()), bins=30)
plt.xlabel('Eigenvector Centrality')
plt.ylabel('Frequency')
plt.title('Eigenvector Centrality Distribution')
plt.show()

betweenness = nx.betweenness_centrality(G)
plt.figure(figsize=(8, 5))
plt.hist(list(betweenness.values()), bins=30)
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')
plt.title('Betweenness Centrality Distribution')
plt.show()

# Assuming 'predictions' is a list of predicted values for each node
node_colors = [f'#{int(p * 255):02x}{int((1-p) * 255):02x}00' for p in predictions]  # Green to red
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color=node_colors, edge_color='gray')
plt.title('Node Predictions Visualization')
plt.show()

# This requires storing embeddings from multiple epochs
# Assuming 'all_embeddings' is a list of embeddings from different epochs
for epoch, embeddings in enumerate(all_embeddings):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    plt.scatter(pca_results[:, 0], pca_results[:, 1], label=f'Epoch {epoch}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Embedding Space Evolution Over Epochs')
plt.legend()
plt.show()

# Assuming you have performance metrics (like accuracy or loss) stored per node degree
degrees = list(performance_by_degree.keys())
performance = list(performance_by_degree.values())

plt.bar(degrees, performance)
plt.xlabel('Node Degree')
plt.ylabel('Performance Metric')
plt.title('Model Performance by Node Degree')
plt.show()

# Assuming 'attention_over_time' is a list of attention matrices, one for each time step
for t, attention_matrix in enumerate(attention_over_time):
    plt.figure(figsize=(6, 6))
    sns.heatmap(attention_matrix, annot=False)
    plt.title(f'Attention Map at Time {t}')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()

def plot_activation_distribution(model, data, layer_indices):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())

    hooks = []
    for i, layer in enumerate(model.children()):
        if i in layer_indices:
            hooks.append(layer.register_forward_hook(hook_fn))

    _ = model(data)
    for hook in hooks:
        hook.remove()

    for i, activation in enumerate(activations):
        plt.figure(figsize=(8, 4))
        plt.hist(activation.numpy().flatten(), bins=30)
        plt.title(f'Activation Distribution in Layer {i}')
        plt.xlabel('Activation')
        plt.ylabel('Frequency')
        plt.show()

# Call this function with the desired layer indices

def visualize_layer_output(model, data, layer_index):
    def hook_fn(module, input, output):
        plt.figure(figsize=(10, 6))
        sns.heatmap(output.detach().cpu().numpy(), annot=False, cmap='viridis')
        plt.title(f'Output of Layer {layer_index}')
        plt.show()

    hook = model[layer_index].register_forward_hook(hook_fn)
    _ = model(data)
    hook.remove()

# Call this function with the desired layer index

# Assuming 'node_features' is a numpy array and 'degrees' is a list of node degrees
plt.figure(figsize=(8, 5))
for i in range(node_features.shape[1]):  # Iterate over features
    plt.scatter(degrees, node_features[:, i], label=f'Feature {i}', alpha=0.7)

plt.xlabel('Node Degree')
plt.ylabel('Feature Value')
plt.title('Correlation between Node Degree and Features')
plt.legend()
plt.show()

# Assuming 'predicted_labels' is an array of predicted labels and 'true_labels' is an array of true labels
unique_labels = np.unique(true_labels)
plt.figure(figsize=(10, 8))
for label in unique_labels:
    plt.hist(predicted_labels[true_labels == label], bins=len(unique_labels), alpha=0.5, label=f'True Label {label}')

plt.xlabel('Predicted Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Labels per Class')
plt.legend()
plt.show()


# Assuming 'attention_weights' is a numpy array of attention weights from your model
# 'node_index' is the index of the node for which to visualize attention
attention = attention_weights[node_index]
plt.figure(figsize=(10, 6))
plt.bar(range(len(attention)), attention)
plt.xlabel('Node Index')
plt.ylabel('Attention Weight')
plt.title(f'Attention Weights for Node {node_index}')
plt.show()

# Assuming 'node_features' and 'node_embeddings' are numpy arrays
feature_index = 0  # Example feature index
embedding_index = 0  # Example embedding dimension

plt.figure(figsize=(8, 5))
plt.scatter(node_features[:, feature_index], node_embeddings[:, embedding_index])
plt.xlabel(f'Feature {feature_index}')
plt.ylabel(f'Embedding Dimension {embedding_index}')
plt.title('Node Feature vs. Embedding Scatter Plot')
plt.show()

# Assuming 'metrics_over_time' stores metrics recorded over time
plt.figure(figsize=(10, 6))
for metric, values in metrics_over_time.items():
    plt.plot(values, label=metric)

plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Training Convergence Plot')
plt.legend()
plt.show()

# Choose two features to compare
feature1_idx, feature2_idx = 0, 1  # Example feature indices

plt.figure(figsize=(8, 8))
plt.scatter(node_features[:, feature1_idx], node_features[:, feature2_idx], c=predictions, cmap='viridis')
plt.xlabel(f'Feature {feature1_idx}')
plt.ylabel(f'Feature {feature2_idx}')
plt.colorbar(label='Prediction')
plt.title('Pairwise Feature Interaction')
plt.show()

# Assuming 'subgraphs' is a list of subgraph node indices
for subgraph in subgraphs:
    sg = G.subgraph(subgraph)
    plt.figure(figsize=(6, 6))
    nx.draw(sg, with_labels=True, node_color='red', edge_color='black')
    plt.title('Subgraph/Motif Visualization')
    plt.show()

# Assuming 'predictions_per_graph' is a list of prediction arrays, one for each graph
plt.figure(figsize=(10, 6))
for i, preds in enumerate(predictions_per_graph):
    sns.kdeplot(preds, label=f'Graph {i}')

plt.xlabel('Predictions')
plt.ylabel('Density')
plt.title('Model\'s Prediction Distribution Over Graphs')
plt.legend()
plt.show()

# Assuming 'layer_responses' is a list where each item contains the response of a layer to input features
for i, response in enumerate(layer_responses):
    plt.figure(figsize=(10, 6))
    sns.heatmap(response, annot=False, cmap='viridis')
    plt.title(f'Layer {i} Response to Input Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Node Index')
    plt.show()

# Assuming 'performance_with_without_sample' stores performance metrics with and without certain training samples
for sample_idx, metrics in performance_with_without_sample.items():
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(metrics)), metrics, label=f'Sample {sample_idx}')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title(f'Impact of Sample {sample_idx} on Learning')
    plt.legend()
    plt.show()

# Assuming 'epochwise_embeddings' is a list of embeddings from each epoch
for epoch, embeddings in enumerate(epochwise_embeddings):
    pca = PCA(n_components=2)
    transformed_embeddings = pca.fit_transform(embeddings)
    plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], label=f'Epoch {epoch}', alpha=0.5)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Evolution of Node Embeddings Across Epochs')
plt.legend()
plt.show()

# Assuming 'performance_metrics' and 'topology_metrics' are dictionaries with the same keys
plt.figure(figsize=(10, 6))
for graph_id, perf_metric in performance_metrics.items():
    topology_metric = topology_metrics[graph_id]
    plt.scatter(topology_metric, perf_metric, label=f'Graph {graph_id}')

plt.xlabel('Graph Topology Metric')
plt.ylabel('Performance Metric')
plt.title('Influence of Graph Topology on Model Performance')
plt.legend()
plt.show()

# Assuming 'cluster_assignments' is a list/array indicating the cluster assignment for each node
for i, cluster in enumerate(np.unique(cluster_assignments)):
    cluster_features = node_features[cluster_assignments == cluster]
    plt.figure(figsize=(8, 5))
    plt.hist(cluster_features.flatten(), bins=30, label=f'Cluster {i}')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title(f'Feature Distribution in Cluster {i}')
    plt.legend()
plt.show()

# Assuming 'edge_predictions' are the predicted values for edges
edge_colors = ['green' if pred > 0.5 else 'red' for pred in edge_predictions]  # Example for binary classification
G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
nx.draw(G, with_labels=True, node_color='skyblue', edge_color=edge_colors)
plt.title('Visualization of Edge Predictions')
plt.show()

# Assuming 'performance_change' records the change in a performance metric after removing each node
plt.figure(figsize=(10, 6))
plt.bar(range(len(performance_change)), performance_change)
plt.xlabel('Node Index')
plt.ylabel('Change in Performance Metric')
plt.title('Effect of Node Removal on Model Performance')
plt.show()

# Assuming 'node_embeddings' is your numpy array of node embeddings
embedding_norms = np.linalg.norm(node_embeddings, axis=1)
plt.figure(figsize=(8, 5))
plt.hist(embedding_norms, bins=30)
plt.xlabel('Norm of Node Embeddings')
plt.ylabel('Frequency')
plt.title('Histogram of Node Embedding Norms')
plt.show()

# Assuming 'model' is your GNN model and 'conv_layer_index' is the index of the convolutional layer
conv_weights = model.conv_layers[conv_layer_index].weight.data.cpu().numpy()
plt.figure(figsize=(8, 8))
sns.heatmap(conv_weights, annot=False, cmap='viridis')
plt.xlabel('Feature Index')
plt.ylabel('Filter Index')
plt.title('Visualization of GNN Filters')
plt.show()

# Assuming 'degrees' is a list/array of node degrees
plt.figure(figsize=(8, 5))
plt.scatter(degrees, embedding_norms)
plt.xlabel('Node Degree')
plt.ylabel('Norm of Node Embedding')
plt.title('Node Degree vs. Embedding Norm')
plt.show()

# Assuming 'activations' is a list where each item contains the activation values of a layer
for i, activation in enumerate(activations):
    plt.figure(figsize=(8, 4))
    plt.hist(activation.flatten(), bins=30)
    plt.xlabel('Activation Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Activation Values in Layer {i}')
    plt.show()

# Assuming 'pre_training_embeddings' and 'post_training_embeddings' are numpy arrays
pca = PCA(n_components=2)
pre_embeddings = pca.fit_transform(pre_training_embeddings)
post_embeddings = pca.transform(post_training_embeddings)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_embeddings[:, 0], pre_embeddings[:, 1])
plt.title('Pre-Training Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(post_embeddings[:, 0], post_embeddings[:, 1])
plt.title('Post-Training Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.show()

for name, param in model.named_parameters():
    if 'weight' in name:
        weights = param.data.cpu().numpy().flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(weights, bins=30)
        plt.xlabel('Weight Values')
        plt.ylabel('Frequency')
        plt.title(f'Weight Distribution in {name}')
        plt.show()

# Assuming you have stored weights from each epoch in 'epoch_weights'
for epoch, weights in enumerate(epoch_weights):
    plt.figure(figsize=(8, 4))
    plt.hist(weights.flatten(), bins=30, alpha=0.5, label=f'Epoch {epoch}')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.title('Evolution of Weights Over Training')
    plt.legend()
    plt.show()

for epoch in range(num_epochs):
    train()
    # Collect gradient norms after backpropagation
    gradient_norms = {name: torch.norm(param.grad).item() for name, param in model.named_parameters() if param.grad is not None}
    
    plt.figure(figsize=(10, 5))
    plt.bar(gradient_norms.keys(), gradient_norms.values())
    plt.xlabel('Layers')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Norms per Layer at Epoch {epoch}')
    plt.xticks(rotation=45)
    plt.show()

# Assuming 'initial_weights' and 'final_weights' store weights from the beginning and end of training
for (initial, final), layer_name in zip(zip(initial_weights, final_weights), model.state_dict()):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(initial.flatten(), bins=30, alpha=0.7)
    plt.title(f'Initial Weights in {layer_name}')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(final.flatten(), bins=30, alpha=0.7)
    plt.title(f'Final Weights in {layer_name}')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')

    plt.show()

from sklearn.cluster import KMeans

# Perform KMeans clustering on node embeddings
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(node_embeddings)

plt.figure(figsize=(8, 8))
plt.scatter(node_embeddings[:, 0], node_embeddings[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.title('Cluster Analysis in Embedding Space')
plt.colorbar(label='Cluster')
plt.show()

# Assuming 'edge_attention_weights' is a list/array of attention weights for each edge
edge_colors = [f'rgba(255,0,0,{w})' for w in edge_attention_weights]  # Red with varying opacity

G = nx.Graph()
G.add_edges_from(data.edge_index.t().tolist())
pos = nx.spring_layout(G)  # Layout for the nodes

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=edge_colors, width=2)
plt.title('Visualization of Attention over Graph Edges')
plt.show()

# Assuming 'loss_values' is a 2D array of loss values sampled from different parameter configurations
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(parameter_range_1, parameter_range_2)  # Parameter ranges
ax.plot_surface(X, Y, loss_values, cmap='viridis', edgecolor='none')
ax.set_xlabel('Parameter 1')
ax.set_ylabel('Parameter 2')
ax.set_zlabel('Loss')
plt.title('Loss Landscape')
plt.show()

# Assuming 'feature_importance_over_time' is a list of feature importance arrays, one for each epoch
for epoch, importance in enumerate(feature_importance_over_time):
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance at Epoch {epoch}')
    plt.show()

# Assuming 'node_performance' and 'node_centrality' are dictionaries with node indices as keys
plt.figure(figsize=(8, 5))
plt.scatter(list(node_centrality.values()), list(node_performance.values()))
plt.xlabel('Node Centrality')
plt.ylabel('Model Performance on Node')
plt.title('Node Centrality vs. Model Performance')
plt.show()

# Assuming 'input_features' and 'output_features' are numpy arrays of the input and output features of the GNN
pca = PCA(n_components=2)
input_pca = pca.fit_transform(input_features)
output_pca = pca.transform(output_features)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(input_pca[:, 0], input_pca[:, 1])
plt.title('Input Feature Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(output_pca[:, 0], output_pca[:, 1])
plt.title('Output Feature Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.show()

# Assuming 'layer_outputs' is a list of outputs from a specific layer for different graphs
for i, output in enumerate(layer_outputs):
    plt.figure(figsize=(8, 6))
    sns.heatmap(output, annot=False, cmap='viridis')
    plt.title(f'Layer Output for Graph {i}')
    plt.xlabel('Output Dimension')
    plt.ylabel('Node Index')
    plt.show()

# Assuming 'node_trajectories' is a list where each element is the embedding of a node at different epochs
for trajectory in node_trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.title('Node Trajectories in Embedding Space Over Time')
plt.show()


correlation_matrix = np.corrcoef(node_features.T, node_embeddings.T)[:node_features.shape[1], node_features.shape[1]:]
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.xlabel('Embedding Dimensions')
plt.ylabel('Feature Dimensions')
plt.title('Correlation Heatmap Between Features and Embeddings')
plt.show()

# Assuming 'layerwise_features' contains the output of each layer for a particular input
for i, features in enumerate(layerwise_features):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.title(f'Feature Space After Layer {i}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Assuming 'layer_outputs' is a list of layer outputs
layer_correlations = np.corrcoef([output.flatten() for output in layer_outputs])
plt.figure(figsize=(8, 6))
sns.heatmap(layer_correlations, annot=True, cmap='coolwarm')
plt.title('Inter-Layer Correlation Heatmap')
plt.xlabel('Layer Index')
plt.ylabel('Layer Index')
plt.show()

# Assuming 'node_importance' is an array of importance scores for each node
plt.figure(figsize=(8, 6))
plt.bar(range(len(node_importance)), node_importance)
plt.xlabel('Node Index')
plt.ylabel('Importance Score')
plt.title('Node Importance Visualization')
plt.show()

# Assuming 'edge_prediction_confidences' is an array of confidence scores for each edge prediction
plt.figure(figsize=(8, 5))
plt.hist(edge_prediction_confidences, bins=30)
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Edge Prediction Confidence Distribution')
plt.show()

# Assuming 'activation_patterns' is an array where each row is an activation pattern for a node
plt.figure(figsize=(10, 8))
sns.heatmap(activation_patterns, annot=False, cmap='viridis')
plt.xlabel('Activation Dimension')
plt.ylabel('Node Index')
plt.title('Activation Pattern Analysis Across Nodes')
plt.show()

# Node degrees, node feature values, and layer weights
degrees = [G.degree(n) for n in G.nodes()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=30, color='green', alpha=0.7)
plt.title("Node Degree Distribution")
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Node feature distribution
feature_index = 0  # Example feature index
plt.figure(figsize=(8, 6))
plt.hist(node_features[:, feature_index], bins=30, color='orange', alpha=0.7)
plt.title(f"Distribution of Node Feature {feature_index}")
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Layer weight distribution
layer_weights = model.layers[0].weight.detach().cpu().numpy().flatten()
plt.figure(figsize=(8, 6))
plt.hist(layer_weights, bins=30, color='blue', alpha=0.7)
plt.title("Weight Distribution of First GNN Layer")
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Assuming 'losses' is a list of recorded losses
window_size = 5
smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(8, 6))
plt.plot(learning_rates[len(learning_rates)-len(smoothed_losses):], smoothed_losses, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Smoothed Loss')
plt.title('Smoothed Learning Rate vs Loss')
plt.xscale('log')
plt.grid(True)
plt.show()

def exponential_moving_average(data, alpha=0.1):
    ema_data = [data[0]]
    for i in range(1, len(data)):
        ema_data.append(alpha * data[i] + (1 - alpha) * ema_data[i-1])
    return ema_data

ema_losses = exponential_moving_average(losses)

plt.figure(figsize=(8, 6))
plt.plot(learning_rates, ema_losses, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('EMA of Loss')
plt.title('Learning Rate vs EMA of Loss')
plt.xscale('log')
plt.grid(True)
plt.show()

batch_learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005]  # Example batch learning rates
batch_losses = [0.45, 0.35, 0.25, 0.30, 0.20]  # Example batch losses

plt.figure(figsize=(8, 6))
plt.scatter(batch_learning_rates, batch_losses, alpha=0.5)
plt.xlabel('Learning Rate')
plt.ylabel('Batch Loss')
plt.title('Batch-wise Learning Rate vs Loss')
plt.xscale('log')
plt.grid(True)
plt.show()

# Simulating a learning rate range test
test_learning_rates = np.logspace(-4, 0, num=100)
test_losses = np.random.rand(100)  # Replace with actual test losses

plt.figure(figsize=(8, 6))
plt.plot(test_learning_rates, test_losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Range Test')
plt.xscale('log')
plt.grid(True)
plt.show()

def visualize_model_activations(model, data, layer_indices):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())

    hooks = []
    for i, layer in enumerate(model.children()):
        if i in layer_indices:
            hooks.append(layer.register_forward_hook(hook_fn))

    _ = model(data)
    for hook in hooks:
        hook.remove()

    for i, activation in enumerate(activations):
        plt.figure(figsize=(10, 6))
        sns.heatmap(activation.cpu().numpy(), annot=False, cmap='viridis')
        plt.title(f'Activation of Layer {i}')
        plt.show()

# Call this function with the desired layer indices

# Assuming 'feature_influence' is a dictionary storing the influence of each feature on model predictions
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_influence)), list(feature_influence.values()), align='center')
plt.xticks(range(len(feature_influence)), list(feature_influence.keys()))
plt.xlabel('Feature')
plt.ylabel('Influence on Predictions')
plt.title('Feature Influence on Model Predictions')
plt.xticks(rotation=45)
plt.show()

# Assuming 'embeddings_over_epochs' stores node embeddings from each epoch
for epoch, embeddings in enumerate(embeddings_over_epochs):
    norms = np.linalg.norm(embeddings, axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(norms, bins=30, alpha=0.5, label=f'Epoch {epoch}')
    plt.xlabel('Norm of Node Embeddings')
    plt.ylabel('Frequency')
    plt.title(f'Embedding Norms Distribution at Epoch {epoch}')
    plt.legend()
    plt.show()

# Assuming 'node_embeddings_over_time' is a list of node embeddings from each epoch
for epoch, embeddings in enumerate(node_embeddings_over_time):
    pca = PCA(n_components=2)
    transformed_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], label=f'Epoch {epoch}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Node Embeddings at Epoch {epoch}')
    plt.legend()
    plt.show()

# Assuming 'predictions_per_graph' is a list of prediction arrays, one for each graph
for i, predictions in enumerate(predictions_per_graph):
    plt.figure(figsize=(8, 5))
    plt.hist(predictions, bins=30, alpha=0.5, label=f'Graph {i}')
    plt.xlabel('Predictions')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Predictions for Graph {i}')
    plt.legend()
    plt.show()

# Assuming 'node_features' is a numpy array of your node features
plt.figure(figsize=(10, 6))
sns.heatmap(node_features, annot=False, cmap='coolwarm')
plt.xlabel('Feature Index')
plt.ylabel('Node Index')
plt.title('Heatmap of Node Features')
plt.show()

# Assuming 'edge_weights' is an array of edge weights from your GNN
plt.figure(figsize=(8, 5))
plt.hist(edge_weights, bins=30)
plt.xlabel('Edge Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Edge Weights')
plt.show()

# Assuming 'performance_metrics' is a dictionary with hyperparameters as keys and performance metrics as values
plt.figure(figsize=(10, 6))
for param, metrics in performance_metrics.items():
    plt.plot(metrics, label=param)

plt.xlabel('Epochs')
plt.ylabel('Performance Metric')
plt.title('Effect of Hyperparameters on Performance')
plt.legend()
plt.show()

# Assuming 'node_features' is a numpy array of node features
correlation_matrix = np.corrcoef(node_features.T)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Node Features')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.show()

# Assuming 'feature_evolution' is a list of features at different training epochs
for epoch, features in enumerate(feature_evolution):
    plt.figure(figsize=(8, 5))
    plt.plot(features)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.title(f'Feature Evolution at Epoch {epoch}')
    plt.show()

# Assuming 'prediction_confidence' is an array of confidence values for model predictions
plt.figure(figsize=(8, 5))
plt.hist(prediction_confidence, bins=30)
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Histogram of Model Prediction Confidence')
plt.show()

