# Computational Complexity Analysis

*Comparison of federated unlearning methods*

| Method | Time Complexity | Space Complexity | Time (s) | Storage (MB) | Speedup | Post-hoc |
|--------|-----------------|------------------|----------|--------------|---------|----------|
| **Ours** | O(\|θ\|) | O(K×\|θ\|) | **0.015** | **5.4** | **123,167×** | ✅ Yes |
| FedEraser | O(T×K×E×\|D\|) | O(K×T/Δt×\|θ\|) | 461.9 | 54 | 4× | Partial |
| FedAU | O(\|θ\|) | O(auxiliary) | 0.001 | 100 | ~10⁶× | ❌ No |
| Retraining | O(T×K×E×\|D\|×\|θ\|) | O(\|θ\|+\|D\|) | 1847.5 | 0 | 1× | ✅ Yes |

## Notes

- K = 10 clients, T = 20 rounds
- SimpleCNN with ~134K parameters
- FedEraser storage assumes Δt = 2
