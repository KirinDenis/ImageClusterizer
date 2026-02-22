## Step 4 — 2D PCA Visualization

**`Polygon\4. ResNet50_2D_PCA_test`**

Reduces high-dimensional vectors (1000D logits or 2048D embeddings) down to 2D for visualization, using Principal Component Analysis via SVD.

### Why PCA?

You can't plot 2048 dimensions. PCA finds the 2 axes along which data varies the most — projecting onto them loses minimum information.

### How it works (step by step):

**1. Build matrix** — stack all vectors into matrix `A` of shape `[N × D]`

**2. Centralize** — subtract column means so the center of mass moves to origin `(0, 0, ...)`
```
centered[i,j] = A[i,j] - mean(A[:,j])
```

**3. SVD decomposition**
```
A = U × ? × V?
```
- `U` — left singular vectors `[N × N]` — projections of each point onto principal axes (normalized)
- `?` (S) — singular values, sorted descending — scale/variance along each axis
- `V?` — right singular vectors — the principal directions themselves (e.g. `[0.707, 0.707, ...]`)

**4. Take first 2 components** — columns 0 and 1 capture the most variance:
```csharp
result[i] = new double[] {
    u[i, 0] * s[0],   // X — projection onto axis of max variance
    u[i, 1] * s[1]    // Y — projection onto axis of second variance
};
```

`U` is normalized (values in `-1..1`), multiply by `S` to restore real scale.

### Singular values tell you how much information is kept:

```
explained_variance = s[0]2 / (s[0]2 + s[1]2 + ... + s[k]2)
```

If first 2 components explain 80%+ — the 2D plot is meaningful. If only 20% — points will look like noise.

### Logits vs Embeddings in PCA:

| | Logits (1000D) | Embeddings (2048D) |
|---|---|---|
| PCA spread | Class-based clusters | Visual/semantic clusters |
| Interpretability | Similar classes group together | Similar looking images group together |
| Info in 2D | Usually moderate | Usually better |

---

## Dependencies

- `MathNet.Numerics` — SVD via `matrix.Svd(computeVectors: true)`
- `Microsoft.ML.OnnxRuntime` — ResNet50 inference
- ResNet50 ONNX model (e.g. from [ONNX Model Zoo](https://github.com/onnx/models))