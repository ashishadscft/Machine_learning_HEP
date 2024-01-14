import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the MLPF_GNN class
class MLPF_GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPF_GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # Pool node features for each graph
        return x

# Example function to train and evaluate the model
def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

    return history

# Define model, criterion, optimizer
input_dim = 3
hidden_dim = 64
output_dim = 5
model = MLPF_GNN(input_dim, hidden_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Update the data creation function for graph-level predictions
def create_dummy_data(num_samples, num_graphs, input_dim, output_dim):
    edge_index = torch.randint(0, num_samples, (2, 1000), dtype=torch.long)
    x = torch.rand((num_samples, input_dim), dtype=torch.float)
    y = torch.randint(0, output_dim, (num_graphs,), dtype=torch.long)  # One label per graph
    batch = torch.randint(0, num_graphs, (num_samples,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, batch=batch)

# Create dataset with correct dimensions
train_data = [create_dummy_data(100, 10, input_dim, output_dim) for _ in range(30)]
val_data = [create_dummy_data(100, 10, input_dim, output_dim) for _ in range(10)]

# Create data loaders for training and validation data
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)

# Train the model
epochs = 10
history = train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs)

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
