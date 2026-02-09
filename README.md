# Federated Unlearning via Lightweight Influence-Aware Reweighting

[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature%20CS-blue)](https://link.springer.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

> **Post-hoc client-level federated unlearning through influence-aware gradient subtraction**

<p align="center">
  <img src="figures/conceptual/fig1_client_influence_decomposition.png" width="80%" alt="Architecture Overview">
</p>

---

## 🎯 Key Contributions

We propose a **lightweight, post-hoc method** for federated unlearning that:

| Feature | Our Method | Benefit |
|---------|------------|---------|
| ✅ **No training modification** | Works on already-trained FL models | Deploy immediately |
| ✅ **123,167× speedup** | 0.015s vs 31 min retraining | Real-time compliance |
| ✅ **5.4 MB storage** | O(K×\|θ\|) complexity | 10× less than FedEraser |
| ✅ **MIA → 51%** | Near random guessing | Effective forgetting |
| ✅ **<2% utility drop** | Preserves model quality | Practical deployment |

---

## 📊 Results at a Glance

### Main Results (α = 0.5)

| Metric | Before | After Unlearning | Retrained Baseline |
|--------|--------|------------------|-------------------|
| **MIA Accuracy** | 68.34 ± 2.15% | **51.23 ± 1.89%** | 50.45 ± 1.12% |
| **Retain Accuracy** | 80.15 ± 1.18% | 78.92 ± 1.52% | 79.23 ± 1.41% |
| **Forget Accuracy** | 78.67 ± 2.34% | 18.45 ± 3.21% | 12.38 ± 2.87% |
| **Cosine Similarity** | - | **0.962 ± 0.005** | 1.000 |
| **Unlearning Time** | - | **0.015s** | 1847.5s |

### Privacy-Utility Trade-off

<p align="center">
  <img src="figures/experiments/fig3_pareto_frontier.png" width="70%" alt="Pareto Frontier">
</p>

---

## 🔬 Method Overview

### Core Formula

```
θᵘ = θᵀ - α × Δθc
```

Where:
- `θᵀ` = Trained federated model
- `Δθc` = Accumulated gradient contribution of client c
- `α` = Unlearning strength parameter (optimal: 0.5)
- `θᵘ` = Resulting unlearned model

### How It Works

1. **During Training**: Track each client's gradient contribution
2. **Upon Request**: Subtract target client's influence with scaling factor α
3. **Result**: Model behaves as if client never participated

<p align="center">
  <img src="figures/conceptual/fig10_gradient_flow_diagram.png" width="80%" alt="Gradient Flow">
</p>

---

## 📁 Repository Structure

```
Federated-Unlearning-Influence-Reweighting/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── requirements.txt             # Python dependencies
│
├── configs/
│   └── experiment.yaml          # Experiment configuration
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── client.py                # FL client implementation
│   ├── server.py                # FL server with gradient tracking
│   └── unlearn.py               # Unlearning algorithm
│
├── scripts/
│   ├── generate_figures.py      # Figure generation
│   └── generate_tables.py       # Table generation
│
├── figures/                     # Publication figures
│   ├── conceptual/              # System diagrams
│   ├── algorithm/               # Method visualizations
│   ├── experiments/             # Empirical results
│   └── comparisons/             # Method comparisons
│
├── results/tables/              # Results in CSV/MD format
│
├── calculations/                # Complexity analysis
│
├── paper/sections/              # LaTeX paper sections
│
├── notebooks/                   # Jupyter notebooks
│
└── docs/                        # Documentation
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/PraneethSiddartha/Federated-Unlearning-Influence-Reweighting.git
cd Federated-Unlearning-Influence-Reweighting

# Create environment
conda create -n fedunlearn python=3.9
conda activate fedunlearn

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Train federated model with gradient tracking
python src/server.py --config configs/experiment.yaml

# Perform unlearning for client 3
python src/unlearn.py --target-client 3 --alpha 0.5
```

---

## 📈 Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | FEMNIST (62 classes) |
| **Model** | SimpleCNN (~134K params) |
| **Clients (K)** | 10 |
| **Rounds (T)** | 20 |
| **Local Epochs (E)** | 5 |
| **Learning Rate** | 0.01 |
| **Non-IID** | Dirichlet α=0.5 |
| **Seeds** | [0, 1, 2, 3, 4] |

---

## 📊 Method Comparison

<p align="center">
  <img src="figures/comparisons/fig7_radar_comparison.png" width="60%" alt="Method Comparison">
</p>

| Method | Speedup | Storage | Post-hoc | Training Mod |
|--------|---------|---------|----------|--------------|
| **Ours** | **123,167×** | **5.4 MB** | ✅ Yes | ❌ No |
| FedEraser | 4× | ~54 MB | Partial | Partial |
| FedAU | ~10⁶× | ~100 MB | ❌ No | ✅ Yes |
| Retraining | 1× | 0 MB | ✅ Yes | ❌ No |

---

## ⚠️ Limitations

1. **Single Dataset**: Evaluated only on FEMNIST
2. **Lightweight Model**: SimpleCNN (~134K params) only
3. **Empirical Guarantees**: Not certified/formal unlearning
4. **Architectural Comparisons**: FedEraser/FedAU comparisons are architectural, not empirical

---

## 📚 Citation

```bibtex
@article{siddartha2026fedunlearn,
  title={Federated Unlearning via Lightweight Influence-Aware Reweighting},
  author={Siddartha, Praneeth},
  journal={SN Computer Science},
  publisher={Springer Nature},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Praneeth Siddartha**

---

<p align="center">
  <b>Target Journal:</b> Springer Nature Computer Science
</p>
