"""
unlearn.py - Unlearning Operations Module
==========================================
Environment: Google Colab + PyTorch + Flower
Author: [Your Name]
Research: Federated Unlearning via Lightweight Influence-Aware Reweighting
Target: Springer Nature Computer Science

Core Operation: θ^u = θ^T - α × Δθ_c

Usage:
    from unlearn import UnlearningModule, simple_mia_attack
    module = UnlearningModule(model)
    unlearned_params, metrics = module.unlearn(params, gradients, alpha=0.5)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import time
import copy


class UnlearningModule:
    """
    Module for executing and validating federated unlearning operations.
    
    Core operation: θ^u = θ^T - α × Δθ_c
    
    Time complexity: O(|θ|) = milliseconds
    Space complexity: O(|θ|) working memory
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def unlearn(
        self,
        model_params: List[np.ndarray],
        client_gradients: List[np.ndarray],
        alpha: float = 0.5,
        clip_norm: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Execute unlearning operation: θ^u = θ^T - α × Δθ_c
        
        Args:
            model_params: Current model parameters θ^T as list of numpy arrays
            client_gradients: Accumulated gradient Δθ_c to subtract
            alpha: Unlearning strength (recommend 0.3-0.7)
            clip_norm: Optional gradient clipping threshold
            
        Returns:
            unlearned_params: θ^u as list of numpy arrays
            metrics: Timing and diagnostic information
        """
        start_time = time.time()
        
        # Validate inputs
        if len(model_params) != len(client_gradients):
            raise ValueError(f"Parameter count mismatch: {len(model_params)} vs {len(client_gradients)}")
        
        # Convert to tensors for efficient operations
        params = [torch.tensor(p, dtype=torch.float32) for p in model_params]
        grads = [torch.tensor(g, dtype=torch.float32) for g in client_gradients]
        
        # Check for NaN/Inf
        for i, g in enumerate(grads):
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"WARNING: NaN/Inf detected in gradient layer {i}, zeroing out")
                grads[i] = torch.zeros_like(g)
        
        # Optional gradient clipping
        original_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads)).item()
        if clip_norm is not None and original_norm > clip_norm:
            scale = clip_norm / original_norm
            grads = [g * scale for g in grads]
            print(f"Gradient clipped: {original_norm:.4f} -> {clip_norm}")
        
        # === CORE UNLEARNING OPERATION ===
        # θ^u = θ^T - α × Δθ_c
        unlearned = [p - alpha * g for p, g in zip(params, grads)]
        
        # Convert back to numpy
        unlearned_params = [u.numpy() for u in unlearned]
        
        elapsed = time.time() - start_time
        grad_norm = original_norm
        param_change = torch.sqrt(sum(torch.sum((u-p)**2) for u, p in zip(unlearned, params))).item()
        
        metrics = {
            "unlearn_time_seconds": elapsed,
            "gradient_norm": grad_norm,
            "parameter_change_norm": param_change,
            "alpha": alpha,
            "clipped": clip_norm is not None and original_norm > clip_norm,
        }
        
        return unlearned_params, metrics
    
    def batch_unlearn(
        self,
        model_params: List[np.ndarray],
        client_gradients_dict: Dict[str, List[np.ndarray]],
        client_ids: List[str],
        alpha: float = 0.5,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Unlearn multiple clients simultaneously.
        
        θ^u = θ^T - α × Σ_{c∈C} Δθ_c
        """
        # Sum all gradients first
        total_grads = None
        valid_clients = []
        
        for cid in client_ids:
            cid = str(cid)
            if cid not in client_gradients_dict:
                print(f"Warning: Client {cid} has no stored gradients, skipping")
                continue
            
            valid_clients.append(cid)
            client_grads = client_gradients_dict[cid]
            
            if total_grads is None:
                total_grads = [np.zeros_like(g) for g in client_grads]
            
            for i, g in enumerate(client_grads):
                total_grads[i] += g
        
        if total_grads is None:
            return model_params, {"error": "No valid clients to unlearn", "clients_unlearned": []}
        
        # Single unlearn operation with combined gradients
        unlearned, metrics = self.unlearn(model_params, total_grads, alpha)
        metrics["clients_unlearned"] = valid_clients
        metrics["num_clients"] = len(valid_clients)
        
        return unlearned, metrics
    
    def find_optimal_alpha(
        self,
        model_params: List[np.ndarray],
        client_gradients: List[np.ndarray],
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        alpha_candidates: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
        lambda_forget: float = 0.5,
    ) -> Tuple[float, Dict]:
        """
        Find optimal α using holdout validation via grid search.
        
        Objective: maximize (retain_acc - λ × forget_acc)
        
        Args:
            model_params: Current model parameters
            client_gradients: Gradients to subtract
            retain_loader: DataLoader for retained client data
            forget_loader: DataLoader for forgotten client data
            alpha_candidates: α values to try
            lambda_forget: Weight for forget penalty
            
        Returns:
            best_alpha: Optimal α value
            results: Dict with results for each α
        """
        results = {}
        
        for alpha in alpha_candidates:
            unlearned_params, _ = self.unlearn(model_params, client_gradients, alpha)
            
            # Load unlearned parameters into model
            self._load_params(unlearned_params)
            
            # Evaluate on retain set (should stay high)
            retain_acc, retain_loss = self._evaluate(retain_loader)
            
            # Evaluate on forget set (should drop)
            forget_acc, forget_loss = self._evaluate(forget_loader)
            
            # Compute score: high retain + low forget = good
            score = retain_acc - lambda_forget * forget_acc
            
            results[alpha] = {
                "retain_acc": retain_acc,
                "retain_loss": retain_loss,
                "forget_acc": forget_acc,
                "forget_loss": forget_loss,
                "score": score,
            }
            
            print(f"α={alpha}: retain_acc={retain_acc:.4f}, forget_acc={forget_acc:.4f}, score={score:.4f}")
        
        # Find best alpha
        best_alpha = max(results.keys(), key=lambda a: results[a]["score"])
        print(f"\nBest α: {best_alpha} (score={results[best_alpha]['score']:.4f})")
        
        return best_alpha, results
    
    def adaptive_alpha_norm_scaled(
        self,
        client_gradients: List[np.ndarray],
        all_client_gradients: Dict[str, List[np.ndarray]],
        alpha_base: float = 0.5,
        alpha_min: float = 0.3,
        alpha_max: float = 0.7,
    ) -> float:
        """
        Compute adaptive α based on gradient norm relative to average.
        
        α_c = α_base × (||Δθ_c|| / avg_norm), clamped to [α_min, α_max]
        """
        # Compute norm for target client
        norm_c = np.sqrt(sum(np.sum(g**2) for g in client_gradients))
        
        # Compute average norm across all clients
        norms = []
        for client_grads in all_client_gradients.values():
            norm = np.sqrt(sum(np.sum(g**2) for g in client_grads))
            norms.append(norm)
        
        avg_norm = np.mean(norms) if norms else norm_c
        
        if avg_norm == 0:
            return alpha_base
        
        alpha = alpha_base * (norm_c / avg_norm)
        return float(np.clip(alpha, alpha_min, alpha_max))
    
    def fine_tune(
        self,
        model_params: List[np.ndarray],
        retain_loader: DataLoader,
        epochs: int = 2,
        lr: float = 0.001,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Optional post-unlearning fine-tuning on retained data.
        
        Use when retain accuracy drops significantly (>2%).
        """
        self._load_params(model_params)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in retain_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(labels)
                total_samples += len(labels)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        fine_tuned_params = [p.cpu().detach().numpy() for p in self.model.parameters()]
        
        return fine_tuned_params, {
            "fine_tune_epochs": epochs,
            "fine_tune_lr": lr,
            "avg_loss": avg_loss,
        }
    
    def _load_params(self, params: List[np.ndarray]):
        """Load numpy parameters into PyTorch model."""
        state_dict = self.model.state_dict()
        param_keys = list(state_dict.keys())
        
        for key, param in zip(param_keys, params):
            state_dict[key] = torch.tensor(param, device=self.device)
        
        self.model.load_state_dict(state_dict)
    
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model accuracy and loss."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * len(labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return accuracy, avg_loss


# === MEMBERSHIP INFERENCE ATTACK ===

def simple_mia_attack(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Simple entropy-based membership inference attack.
    
    Intuition: Training members have lower entropy (higher confidence).
    After successful unlearning, MIA accuracy should be ~50% (random guess).
    
    Args:
        model: Target model to attack
        member_loader: DataLoader with training data (should be identified as members)
        nonmember_loader: DataLoader with non-training data
        device: Torch device
        
    Returns:
        Dict with attack metrics:
        - mia_accuracy: Should be ~50% after unlearning
        - mia_auc: Area under ROC curve
        - threshold: Entropy threshold used
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    def get_entropy(loader):
        entropies = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                outputs = torch.softmax(model(images), dim=1)
                # Entropy: -Σ p_i * log(p_i)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)
                entropies.extend(entropy.cpu().numpy())
        return np.array(entropies)
    
    member_entropy = get_entropy(member_loader)
    nonmember_entropy = get_entropy(nonmember_loader)
    
    if len(member_entropy) == 0 or len(nonmember_entropy) == 0:
        return {"mia_accuracy": 0.5, "mia_auc": 0.5, "error": "Empty loaders"}
    
    # Members typically have lower entropy (higher confidence)
    # Use median entropy as threshold
    all_entropy = np.concatenate([member_entropy, nonmember_entropy])
    threshold = np.median(all_entropy)
    
    # Predict: entropy < threshold → member
    member_preds = member_entropy < threshold  # True = predicted as member
    nonmember_preds = nonmember_entropy < threshold
    
    # Calculate metrics
    tp = np.sum(member_preds)  # Correctly identified members
    fn = len(member_preds) - tp  # Missed members
    fp = np.sum(nonmember_preds)  # False positives
    tn = len(nonmember_preds) - fp  # True negatives
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.5
    
    # AUC calculation
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.ones(len(member_entropy)), np.zeros(len(nonmember_entropy))])
        scores = np.concatenate([-member_entropy, -nonmember_entropy])  # Negative because lower entropy = member
        auc = roc_auc_score(labels, scores)
    except ImportError:
        # Fallback if sklearn not available
        auc = accuracy  # Rough approximation
    
    return {
        "mia_accuracy": float(accuracy),
        "mia_auc": float(auc),
        "threshold": float(threshold),
        "member_entropy_mean": float(np.mean(member_entropy)),
        "nonmember_entropy_mean": float(np.mean(nonmember_entropy)),
        "num_members": len(member_entropy),
        "num_nonmembers": len(nonmember_entropy),
    }


def shadow_model_mia(
    target_model: nn.Module,
    target_loader: DataLoader,
    shadow_member_loader: DataLoader,
    shadow_nonmember_loader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Shadow model-based MIA (more sophisticated).
    
    Trains an attack model on shadow data, then applies to target.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_confidence_features(model, loader):
        """Extract prediction confidence as features."""
        model.eval()
        features = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                outputs = torch.softmax(model(images), dim=1)
                # Features: top-k confidences, entropy, max confidence
                topk, _ = outputs.topk(3, dim=1)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1, keepdim=True)
                feat = torch.cat([topk, entropy, outputs.max(1, keepdim=True)[0]], dim=1)
                features.append(feat.cpu().numpy())
        return np.vstack(features)
    
    # Get features for shadow data
    shadow_member_feat = get_confidence_features(target_model, shadow_member_loader)
    shadow_nonmember_feat = get_confidence_features(target_model, shadow_nonmember_loader)
    
    # Train attack classifier
    from sklearn.linear_model import LogisticRegression
    X_train = np.vstack([shadow_member_feat, shadow_nonmember_feat])
    y_train = np.concatenate([np.ones(len(shadow_member_feat)), np.zeros(len(shadow_nonmember_feat))])
    
    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(X_train, y_train)
    
    # Apply to target
    target_feat = get_confidence_features(target_model, target_loader)
    predictions = attack_model.predict(target_feat)
    probabilities = attack_model.predict_proba(target_feat)[:, 1]
    
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "attack_model": attack_model,
    }


# === UTILITY FUNCTIONS ===

def compute_cosine_similarity(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Compute cosine similarity between two parameter sets."""
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    
    dot = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    return float(dot / (norm1 * norm2 + 1e-10))


def compute_l2_distance(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Compute L2 distance between two parameter sets."""
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    return float(np.linalg.norm(flat1 - flat2))


def validate_unlearning(
    original_params: List[np.ndarray],
    unlearned_params: List[np.ndarray],
    retrained_params: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Validate unlearning quality by comparing parameters.
    
    Returns metrics indicating how well unlearning approximates retraining.
    """
    metrics = {
        "param_change_norm": compute_l2_distance(original_params, unlearned_params),
        "param_change_relative": compute_l2_distance(original_params, unlearned_params) / 
                                  (np.linalg.norm(np.concatenate([p.flatten() for p in original_params])) + 1e-10),
    }
    
    if retrained_params is not None:
        metrics["cosine_sim_to_retrained"] = compute_cosine_similarity(unlearned_params, retrained_params)
        metrics["l2_dist_to_retrained"] = compute_l2_distance(unlearned_params, retrained_params)
    
    return metrics


# === GRADIENT COMPRESSION ===

def quantize_8bit(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Quantize float32 tensor to int8 for storage efficiency.
    
    Compression: 4×
    Error: < 0.5%
    """
    abs_max = np.abs(tensor).max()
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    
    scale = 127.0 / abs_max
    quantized = np.round(tensor * scale).astype(np.int8)
    return quantized, float(abs_max)


def dequantize_8bit(quantized: np.ndarray, abs_max: float) -> np.ndarray:
    """Reconstruct float32 tensor from int8."""
    return quantized.astype(np.float32) * (abs_max / 127.0)


def topk_sparsify(tensor: np.ndarray, k_percent: float = 1.0) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Keep only top k% of values by magnitude.
    
    k=1%: ~50× compression
    k=5%: ~10× compression
    """
    shape = tensor.shape
    flat = tensor.flatten()
    k = max(1, int(len(flat) * k_percent / 100))
    
    top_indices = np.argsort(np.abs(flat))[-k:]
    top_values = flat[top_indices].astype(np.float32)
    
    return top_indices.astype(np.int32), top_values, shape


def reconstruct_topk(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct tensor from sparse representation."""
    flat = np.zeros(np.prod(shape), dtype=np.float32)
    flat[indices] = values
    return flat.reshape(shape)


# === ENTRY POINT ===
if __name__ == "__main__":
    print("=" * 60)
    print("Unlearning Module - Federated Unlearning Operations")
    print("=" * 60)
    print("Core operation: θ^u = θ^T - α × Δθ_c")
    print()
    
    # Test with dummy data
    print("Running self-test with dummy data...")
    
    dummy_params = [np.random.randn(100, 100).astype(np.float32) for _ in range(3)]
    dummy_grads = [np.random.randn(100, 100).astype(np.float32) * 0.1 for _ in range(3)]
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
            )
        def forward(self, x):
            return self.layers(x)
    
    module = UnlearningModule(DummyModel())
    unlearned, metrics = module.unlearn(dummy_params, dummy_grads, alpha=0.5)
    
    print(f"✓ Unlearning completed in {metrics['unlearn_time_seconds']:.6f}s")
    print(f"✓ Gradient norm: {metrics['gradient_norm']:.4f}")
    print(f"✓ Parameter change: {metrics['parameter_change_norm']:.4f}")
    print()
    print("Self-test passed! Module ready for use.")
