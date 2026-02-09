"""
server.py - Federated Learning Server with Unlearning Support
============================================================
Environment: Google Colab + PyTorch + Flower
Author: [Your Name]
Research: Federated Unlearning via Lightweight Influence-Aware Reweighting
Target: Springer Nature Computer Science

Usage:
    from server import FedAvgWithUnlearning, create_strategy
    strategy = create_strategy()
"""

import flwr as fl
from flwr.common import (
    Parameters, 
    Scalar, 
    ndarrays_to_parameters, 
    parameters_to_ndarrays,
    FitRes,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import time
from collections import defaultdict
import os

# === CONFIGURATION ===
CONFIG = {
    "num_clients": 10,
    "clients_per_round": 5,
    "num_rounds": 20,
    "unlearning_alpha": 0.5,
    "gradient_clip_norm": None,  # Set to float for clipping (e.g., 1.0)
    "results_dir": "results/",
}


class FedAvgWithUnlearning(FedAvg):
    """
    Extended FedAvg strategy that accumulates per-client gradients
    for later unlearning operations.
    
    Core operation: θ^u = θ^T - α × Δθ_c
    
    Storage: O(K × |θ|) where K = num_clients, |θ| = model size
    Unlearning time: O(|θ|) = milliseconds
    """
    
    def __init__(
        self,
        unlearning_alpha: float = 0.5,
        gradient_clip_norm: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unlearning_alpha = unlearning_alpha
        self.gradient_clip_norm = gradient_clip_norm
        
        # === GRADIENT STORAGE ===
        # Maps client_id -> accumulated weighted gradient
        self.client_accumulated_gradients: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        # Temporary storage for computing deltas
        self.round_initial_params: Optional[List[np.ndarray]] = None
        self.current_round: int = 0
        
        # Metadata tracking
        self.client_participation: Dict[str, List[int]] = defaultdict(list)
        self.client_samples: Dict[str, int] = defaultdict(int)
        
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """
        Called before each round to configure client training.
        HOOK: Store θ^{t-1} for delta computation.
        """
        self.current_round = server_round
        self.round_initial_params = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates AND accumulate gradients for unlearning.
        
        CRITICAL HOOK: This is where gradient accumulation happens.
        """
        if self.round_initial_params is None:
            return super().aggregate_fit(server_round, results, failures)
        
        # === ACCUMULATE GRADIENTS FOR UNLEARNING ===
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        
        for client_proxy, fit_res in results:
            # Get client ID from metrics (MUST be set by client)
            client_id = str(fit_res.metrics.get("client_id", client_proxy.cid))
            client_params = parameters_to_ndarrays(fit_res.parameters)
            client_samples = fit_res.num_examples
            
            # Compute Δθ_k^t = θ_k^t - θ^{t-1}
            delta_theta = [
                c - i for c, i in zip(client_params, self.round_initial_params)
            ]
            
            # Weight by data proportion: w_k = n_k / Σn_j
            weight = client_samples / total_samples if total_samples > 0 else 1.0 / len(results)
            weighted_delta = [weight * d for d in delta_theta]
            
            # Initialize if first participation
            if client_id not in self.client_accumulated_gradients or \
               len(self.client_accumulated_gradients[client_id]) == 0:
                self.client_accumulated_gradients[client_id] = [
                    np.zeros_like(d) for d in weighted_delta
                ]
            
            # Accumulate: ΔΘ[k] += w_k × Δθ_k^t
            for i, wd in enumerate(weighted_delta):
                self.client_accumulated_gradients[client_id][i] += wd
            
            # Track participation
            self.client_participation[client_id].append(server_round)
            self.client_samples[client_id] = client_samples
        
        # Proceed with standard FedAvg aggregation
        return super().aggregate_fit(server_round, results, failures)
    
    def unlearn_clients(
        self,
        current_params: List[np.ndarray],
        client_ids: List[str],
        alpha: Optional[float] = None
    ) -> Tuple[List[np.ndarray], Dict[str, any]]:
        """
        Execute unlearning: θ^u = θ^T - α × Σ_{c∈C} Δθ_c
        
        Args:
            current_params: Current model parameters θ^T as list of numpy arrays
            client_ids: List of client IDs to unlearn
            alpha: Unlearning strength (default: self.unlearning_alpha)
        
        Returns:
            unlearned_params: New model parameters θ^u
            metrics: Dict with timing and diagnostic information
        """
        start_time = time.time()
        alpha = alpha if alpha is not None else self.unlearning_alpha
        
        # Initialize total subtraction
        total_delta = [np.zeros_like(p) for p in current_params]
        skipped_clients = []
        unlearned_clients = []
        
        for client_id in client_ids:
            client_id = str(client_id)
            
            if client_id not in self.client_accumulated_gradients or \
               len(self.client_accumulated_gradients[client_id]) == 0:
                skipped_clients.append(client_id)
                print(f"WARNING: Client {client_id} has no stored gradients, skipping")
                continue
            
            client_delta = self.client_accumulated_gradients[client_id]
            
            # Optional: Gradient clipping for stability
            if self.gradient_clip_norm is not None:
                delta_norm = np.sqrt(sum(np.sum(d**2) for d in client_delta))
                if delta_norm > self.gradient_clip_norm:
                    scale = self.gradient_clip_norm / delta_norm
                    client_delta = [d * scale for d in client_delta]
                    print(f"Clipped gradient for client {client_id}: {delta_norm:.4f} -> {self.gradient_clip_norm}")
            
            # Accumulate for batch deletion
            for i, d in enumerate(client_delta):
                total_delta[i] += d
            
            unlearned_clients.append(client_id)
        
        # === CORE UNLEARNING OPERATION ===
        # θ^u = θ^T - α × Σ Δθ_c
        unlearned_params = [p - alpha * d for p, d in zip(current_params, total_delta)]
        
        elapsed = time.time() - start_time
        delta_norm = np.sqrt(sum(np.sum(d**2) for d in total_delta))
        param_change = np.sqrt(sum(np.sum((u-p)**2) for u, p in zip(unlearned_params, current_params)))
        
        metrics = {
            "unlearn_time_seconds": elapsed,
            "total_delta_norm": float(delta_norm),
            "param_change_norm": float(param_change),
            "alpha_used": alpha,
            "clients_unlearned": unlearned_clients,
            "clients_skipped": skipped_clients,
            "num_unlearned": len(unlearned_clients),
            "num_skipped": len(skipped_clients),
        }
        
        print(f"Unlearning completed in {elapsed:.4f}s")
        print(f"  - Clients unlearned: {unlearned_clients}")
        print(f"  - Alpha: {alpha}, Delta norm: {delta_norm:.4f}")
        
        return unlearned_params, metrics
    
    def get_client_info(self) -> Dict[str, Dict]:
        """Get information about all tracked clients."""
        info = {}
        for client_id in self.client_accumulated_gradients:
            delta = self.client_accumulated_gradients[client_id]
            norm = np.sqrt(sum(np.sum(d**2) for d in delta)) if delta else 0
            info[client_id] = {
                "gradient_norm": float(norm),
                "participation_rounds": self.client_participation.get(client_id, []),
                "num_samples": self.client_samples.get(client_id, 0),
            }
        return info
    
    def save_gradient_store(self, filepath: str):
        """Save accumulated gradients and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "gradients": dict(self.client_accumulated_gradients),
            "participation": dict(self.client_participation),
            "samples": dict(self.client_samples),
            "config": {
                "unlearning_alpha": self.unlearning_alpha,
                "gradient_clip_norm": self.gradient_clip_norm,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved gradient store to {filepath}")
        print(f"  - Clients tracked: {len(self.client_accumulated_gradients)}")
    
    def load_gradient_store(self, filepath: str):
        """Load accumulated gradients and metadata from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.client_accumulated_gradients = defaultdict(list, data["gradients"])
        self.client_participation = defaultdict(list, data.get("participation", {}))
        self.client_samples = defaultdict(int, data.get("samples", {}))
        
        print(f"Loaded gradient store from {filepath}")
        print(f"  - Clients loaded: {len(self.client_accumulated_gradients)}")


def create_strategy(**kwargs) -> FedAvgWithUnlearning:
    """Factory function to create the FL strategy with default config."""
    default_kwargs = {
        "fraction_fit": CONFIG["clients_per_round"] / CONFIG["num_clients"],
        "min_fit_clients": CONFIG["clients_per_round"],
        "min_available_clients": CONFIG["num_clients"],
        "unlearning_alpha": CONFIG["unlearning_alpha"],
        "gradient_clip_norm": CONFIG["gradient_clip_norm"],
    }
    default_kwargs.update(kwargs)
    return FedAvgWithUnlearning(**default_kwargs)


# === UTILITY FUNCTIONS ===
def compute_cosine_similarity(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Compute cosine similarity between two parameter sets."""
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    dot = np.dot(flat1, flat2)
    norm1, norm2 = np.linalg.norm(flat1), np.linalg.norm(flat2)
    return float(dot / (norm1 * norm2 + 1e-10))


def compute_l2_distance(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    """Compute L2 distance between two parameter sets."""
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    return float(np.linalg.norm(flat1 - flat2))


# === ENTRY POINT ===
if __name__ == "__main__":
    print("=" * 60)
    print("Federated Learning Server with Unlearning Support")
    print("=" * 60)
    print(f"Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()
    
    strategy = create_strategy()
    print("Strategy created successfully!")
    print("Ready for FL training with gradient accumulation.")
