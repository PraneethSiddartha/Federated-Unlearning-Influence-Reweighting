# Computational Complexity Analysis

## Time Complexity

### Our Method: O(|θ|)

The unlearning operation is a single vector subtraction:

```
θᵘ = θᵀ - α × Δθc
```

- θᵀ has |θ| = 134,590 parameters
- Single subtraction operation
- **Measured time: 0.015 seconds**

### Retraining: O(T × K × E × |D| × |θ|)

Full retraining requires:
- T = 20 communication rounds
- K = 10 clients participating
- E = 5 local epochs per round
- |D| = dataset size per client
- **Measured time: 1847.5 seconds**

### Speedup Calculation

```
Speedup = Time_retrain / Time_ours
        = 1847.5 / 0.015
        = 123,166.67
        ≈ 123,167×
```

---

## Space Complexity

### Our Method: O(K × |θ|)

Storage requirement:
- Store Δθc for each of K clients
- Each Δθc has |θ| parameters

```
Storage = K × |θ| × 4 bytes (float32)
        = 10 × 134,590 × 4
        = 5,383,600 bytes
        ≈ 5.4 MB
```

### FedEraser: O(K × T/Δt × |θ|)

Stores historical checkpoints:
```
Storage = K × (T/Δt) × |θ| × 4 bytes
        = 10 × (20/2) × 134,590 × 4
        = 53,836,000 bytes
        ≈ 54 MB (with Δt=2)
```

### Storage Comparison

| Method | Formula | Concrete Value |
|--------|---------|----------------|
| Ours | K × \|θ\| × 4 | 5.4 MB |
| FedEraser (Δt=2) | K × T/Δt × \|θ\| × 4 | 54 MB |
| Retraining | 0 additional | 0 MB |

**Our method uses 10× less storage than FedEraser.**
