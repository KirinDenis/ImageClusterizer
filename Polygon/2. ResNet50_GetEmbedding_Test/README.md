## Step 2 — Feature Extraction (Embedding vectors)

**`Polygon\2. ResNet50_GetEmbedding_Test`**

Instead of taking the final classification layer output, we extract the penultimate layer — the **embedding vector** (also called feature vector). This is a dense representation of the image content, not tied to any specific class.

```
Input image › ResNet50 › [remove final FC layer] › float[2048] embedding
```

- Dimension = 2048 (ResNet50 average pooling layer output)
- Captures semantic meaning of the image
- Used for similarity search, clustering, transfer learning

---
