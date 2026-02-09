# Storage Overhead Analysis

## SimpleCNN Architecture

| Layer | Parameters |
|-------|------------|
| Conv1 (1→32, 5×5) | 832 |
| Conv2 (32→64, 5×5) | 51,264 |
| FC1 (1024→512) | 524,800 |
| FC2 (512→62) | 31,806 |
| **Total** | **~134,590** |

## Storage Requirements

### Per-Client Storage

```
Δθc storage = |θ| × sizeof(float32)
            = 134,590 × 4 bytes
            = 538,360 bytes
            ≈ 0.54 MB per client
```

### Total Storage (K=10 clients)

```
Total = K × 0.54 MB
      = 10 × 0.54 MB
      = 5.4 MB
```

## Comparison with Baselines

| Method | Storage Formula | K=10, SimpleCNN |
|--------|-----------------|-----------------|
| **Ours** | K × \|θ\| × 4 | **5.4 MB** |
| FedEraser (Δt=2) | K × T/Δt × \|θ\| × 4 | 54 MB |
| FedEraser (Δt=5) | K × T/Δt × \|θ\| × 4 | 21.6 MB |
| Retraining | 0 | 0 MB |

## Scaling Analysis

| Clients (K) | Our Method | FedEraser (Δt=2) |
|-------------|------------|------------------|
| 10 | 5.4 MB | 54 MB |
| 50 | 27 MB | 270 MB |
| 100 | 54 MB | 540 MB |
