## Step 3 — Cosine Similarity Search

**`Polygon\3. ResNet50_Image_similarity_search_test`**

Compares images by measuring the angle between their vectors in high-dimensional space. Works for both logits (1000D) and embeddings (2048D).

```
CosineSimilarity(A, B) = (A · B) / (|A| * |B|)
```

- Result range: `-1` (opposite) › `0` (unrelated) › `1` (identical)
- Not sensitive to vector magnitude, only direction
- Embeddings generally give better similarity results than logits
- Logits similarity reflects class-level similarity, embeddings reflect visual similarity

---
