## Step 5. Image Similarity with Sparse Embeddings

**`Polygon\5. ResNet50_Sparse_Dot_Product_test`**

Demonstration of sparse vector representation for image similarity search using ResNet50 embeddings. Shows how to reduce 2048-dimensional vectors to sparse format (top-N values) and compute cosine similarity efficiently.

### What this does

1. **Extract embeddings** from images using ResNet50 (2048-dimensional vectors)
2. **Convert to sparse representation** - keep only top-10 values with their positions
3. **Compute sparse cosine similarity** between images using reduced vectors

This approach trades some accuracy for significant memory savings and faster comparisons.

### How sparse vectors work

**Full embedding** (2048 dimensions):
```
[0.12, -0.45, 0.89, 0.03, ..., 0.67, 0.22]  // 2048 floats
```

**Sparse embedding** (top-10 values only):
```csharp
Dictionary<int, float> {
    { 342, 0.89 },   // position 342 had value 0.89
    { 1024, 0.67 },  // position 1024 had value 0.67
    { 567, 0.55 },
    // ... 7 more
}
```

Memory: **2048 × 4 bytes = 8 KB** → **10 × 12 bytes = 120 bytes** (67x smaller)

### Example output
```
12847 files found - 1234 ms
vectors complete for 12847 image files - 145623 ms
sparse vectors calculated - 89 ms

Embedding cat.4001.copy.jpg = 0.9987  ← Nearly identical
Embedding cat.4002.jpg = 0.8234       ← Similar cat
Embedding dog.4081.jpg = 0.3421       ← Different animal
Embedding noise1.jpg = 0.0456         ← Random noise

dot product first image to all completed - 234 ms