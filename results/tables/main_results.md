# Main Unlearning Results

*Target: Client 3, α = 0.5, averaged over 5 seeds*

| Condition | Retain Acc (%) | Forget Acc (%) | MIA Acc (%) | Time (s) |
|-----------|----------------|----------------|-------------|----------|
| Before Unlearning | 80.15 ± 1.18 | 78.67 ± 2.34 | 68.34 ± 2.15 | - |
| **After Unlearning** | **78.92 ± 1.52** | **18.45 ± 3.21** | **51.23 ± 1.89** | **0.015** |
| Retrained Baseline | 79.23 ± 1.41 | 12.38 ± 2.87 | 50.45 ± 1.12 | 1847.5 |

## Key Findings

- **MIA drops to near-random**: 68.34% → 51.23% (random = 50%)
- **Utility preserved**: Only 1.23% accuracy drop
- **Speedup**: 123,167× faster than retraining
