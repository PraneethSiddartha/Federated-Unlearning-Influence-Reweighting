"""
client.py - Federated Learning Client with Unlearning Support
=============================================================
Environment: Google Colab + PyTorch + Flower
Author: [Your Name]
Research: Federated Unlearning via Lightweight Influence-Aware Reweighting
Target: Springer Nature Computer Science

Usage:
    from client import client_fn, SimpleCNN
    client = client_fn("0")  # Create client with ID "0"
"""

import flwr as fl
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Dict, List, Tuple, Callable
import numpy as np

# === CONFIGURATION ===
CLIENT_CONFIG = {
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# === MODEL DEFINITIONS ===

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/FEMNIST (28x28 grayscale images).
    Parameters: ~100K (suitable for Colab)
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 (32x32 RGB images).
    Parameters: ~500K
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 32->16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 16->8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 8->4
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    """
    MLP for tabular data (e.g., Credit Card Fraud).
    Parameters: ~20K
    """
    
    def __init__(self, input_dim: int = 30, num_classes: int = 2, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# === PARAMETER UTILITIES ===

def get_parameters(model: nn.Module) -> NDArrays:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Load parameters into model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# === FLOWER CLIENT ===

class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning with unlearning support.
    
    CRITICAL: Returns client_id in fit() metrics for gradient tracking.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
    ):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model.to(device)
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters."""
        return get_parameters(self.model)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Local training on client's private data.
        
        CRITICAL: Returns client_id in metrics for server-side gradient tracking.
        
        Returns:
            parameters: Updated model parameters (θ_k^t)
            num_examples: Number of training samples
            metrics: Dict including client_id for gradient tracking
        """
        # Load global model parameters
        set_parameters(self.model, parameters)
        
        # Get training config (can be overridden by server)
        epochs = config.get("local_epochs", CLIENT_CONFIG["local_epochs"])
        lr = config.get("lr", CLIENT_CONFIG["learning_rate"])
        
        # Local training
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=CLIENT_CONFIG["momentum"]
        )
        criterion = nn.CrossEntropyLoss()
        
        total_samples = 0
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in self.trainloader:
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch[0], batch[1]
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(labels)
                total_samples += len(labels)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        samples_per_epoch = total_samples // epochs if epochs > 0 else total_samples
        
        # CRITICAL: Include client_id in metrics for gradient tracking!
        metrics = {
            "client_id": self.client_id,
            "train_loss": avg_loss,
            "epochs": epochs,
        }
        
        return get_parameters(self.model), samples_per_epoch, metrics
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local test data."""
        set_parameters(self.model, parameters)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.testloader:
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch[0], batch[1]
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return avg_loss, total, {
            "accuracy": accuracy, 
            "client_id": self.client_id
        }


# === DATA PARTITIONING ===

def partition_data_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    IID data partitioning: each client gets random subset.
    
    Returns:
        Dict mapping client_id -> list of sample indices
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, num_clients)
    return {i: list(split_indices[i]) for i in range(num_clients)}


def partition_data_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Non-IID data partitioning using Dirichlet distribution.
    
    Args:
        alpha: Dirichlet concentration parameter
               - alpha = 0.1: highly non-IID
               - alpha = 1.0: moderately non-IID  
               - alpha = 10.0: nearly IID
    
    Returns:
        Dict mapping client_id -> list of sample indices
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Fallback: iterate through dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Assign indices to clients based on proportions
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()  # Fix rounding
        
        start = 0
        for client_id, num_samples in enumerate(proportions):
            client_indices[client_id].extend(
                class_indices[start:start + num_samples].tolist()
            )
            start += num_samples
    
    # Shuffle each client's indices
    for client_id in client_indices:
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


# === CLIENT FACTORY ===

def client_fn(
    cid: str,
    model_fn: Callable[[], nn.Module] = None,
    dataset: Dataset = None,
    partition: Dict[int, List[int]] = None,
) -> FlowerClient:
    """
    Factory function to create a Flower client.
    
    Args:
        cid: Client ID (string)
        model_fn: Function that returns a model instance
        dataset: Full training dataset
        partition: Dict mapping client_id -> sample indices
    
    For Colab usage, set these globally or pass via closure.
    """
    device = torch.device(CLIENT_CONFIG["device"])
    
    # Default model if not provided
    if model_fn is None:
        model_fn = lambda: SimpleCNN(num_classes=10)
    
    model = model_fn()
    
    # === DATA LOADING ===
    # If dataset and partition provided, use them
    if dataset is not None and partition is not None:
        client_idx = int(cid)
        if client_idx in partition:
            indices = partition[client_idx]
        else:
            indices = partition[str(client_idx)]
        
        client_dataset = Subset(dataset, indices)
    else:
        # PLACEHOLDER: Replace with actual data loading
        # This creates dummy data for testing
        print(f"WARNING: Using dummy data for client {cid}")
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
        # Simple partition for testing
        num_clients = 10
        client_idx = int(cid) % num_clients
        samples_per_client = len(full_dataset) // num_clients
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client
        client_dataset = Subset(full_dataset, range(start_idx, end_idx))
    
    # Create data loaders
    trainloader = DataLoader(
        client_dataset, 
        batch_size=CLIENT_CONFIG["batch_size"], 
        shuffle=True,
        num_workers=0,  # For Colab compatibility
    )
    
    # Use subset for local evaluation
    eval_size = min(500, len(client_dataset))
    eval_indices = list(range(eval_size))
    testloader = DataLoader(
        Subset(client_dataset, eval_indices),
        batch_size=CLIENT_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    return FlowerClient(
        client_id=cid,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
    )


# === ENTRY POINT ===
if __name__ == "__main__":
    print("=" * 60)
    print("Federated Learning Client")
    print("=" * 60)
    print(f"Configuration:")
    for k, v in CLIENT_CONFIG.items():
        print(f"  {k}: {v}")
    print()
    
    # Test client creation
    client = client_fn("0")
    print(f"Client 0 created successfully!")
    print(f"  - Training samples: {len(client.trainloader.dataset)}")
    print(f"  - Model parameters: {sum(p.numel() for p in client.model.parameters()):,}")
