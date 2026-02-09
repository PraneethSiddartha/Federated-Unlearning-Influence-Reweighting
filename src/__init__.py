"""
Federated Unlearning via Lightweight Influence-Aware Reweighting
================================================================

Post-hoc client-level unlearning using influence-aware gradient subtraction.

Core Formula: θᵘ = θᵀ - α × Δθc

Modules:
    - client: Federated learning client
    - server: FL server with gradient tracking
    - unlearn: Unlearning algorithm
"""

__version__ = "1.0.0"
__author__ = "Praneeth Siddartha"
