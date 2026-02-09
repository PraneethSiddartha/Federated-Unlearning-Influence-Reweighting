# α Ablation Study Results

*Effect of unlearning strength parameter α on privacy-utility trade-off*

| α | Retain Acc (%) | Forget Acc (%) | MIA Acc (%) | Cosine Sim |
|---|----------------|----------------|-------------|------------|
| 0.3 | 79.85 ± 1.21 | 45.32 ± 4.12 | 58.67 ± 2.45 | 0.987 ± 0.003 |
| 0.4 | 79.12 ± 1.34 | 31.45 ± 3.87 | 54.23 ± 2.12 | 0.978 ± 0.004 |
| **0.5** | **78.92 ± 1.52** | **18.45 ± 3.21** | **51.23 ± 1.89** | **0.962 ± 0.005** |
| 0.6 | 76.45 ± 1.78 | 12.78 ± 2.95 | 50.45 ± 1.56 | 0.941 ± 0.006 |
| 0.7 | 72.34 ± 2.12 | 8.92 ± 2.34 | 49.87 ± 1.23 | 0.912 ± 0.008 |

## Analysis

- **Optimal α = 0.5**: Best privacy-utility balance
- Higher α → Better privacy (lower MIA) but worse utility
- Lower α → Better utility but worse privacy
