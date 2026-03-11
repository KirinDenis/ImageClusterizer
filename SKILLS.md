# ImageClusterizer — Build from Scratch: Skills & Architecture Guide

> **Branch:** `dev` | **Target:** .NET 8 · WPF · ONNX Runtime · ResNet-50 · MVVM
> **Purpose:** A complete skill map and implementation guide so any developer can reproduce this project from zero.

---

## Table of Contents

1. [Prerequisites & Toolchain](#1-prerequisites--toolchain)
2. [Solution Structure & MVVM Architecture](#2-solution-structure--mvvm-architecture)
3. [NuGet Packages](#3-nuget-packages)
4. [ResNet-50 & ONNX Integration](#4-resnet-50--onnx-integration)
5. [Image Preprocessing Pipeline](#5-image-preprocessing-pipeline)
6. [Embedding Vectors — Extraction & Storage](#6-embedding-vectors--extraction--storage)
7. [Cosine Similarity — Fast Implementation](#7-cosine-similarity--fast-implementation)
8. [PCA via Randomized SVD (RSVD/Halko)](#8-pca-via-randomized-svd-rsvdhalko)
9. [SQLite Caching Layer](#9-sqlite-caching-layer)
10. [Clustering Pipeline](#10-clustering-pipeline)
11. [WPF UI Patterns](#11-wpf-ui-patterns)
12. [Polygon — Research Sandbox](#12-polygon--research-sandbox)
13. [Performance Checklist](#13-performance-checklist)
14. [Build & Run from Zero](#14-build--run-from-zero)

---

## 1. Prerequisites & Toolchain

| Tool | Version | Purpose |
|------|---------|---------|
| Visual Studio 2022 | 17.9+ | IDE with WPF designer |
| .NET SDK | 8.0 | Runtime & build system |
| Windows | 10/11 x64 | WPF requires Windows |
| Git | any | Version control |
| Python *(optional)* | 3.10+ | For Polygon research notebooks |

**Install checklist:**
- Install Visual Studio 2022 with workloads: `.NET desktop development`
- Install .NET 8 SDK: https://dotnet.microsoft.com/download/dotnet/8.0
- Clone repo: `git clone https://github.com/KirinDenis/ImageClusterizer.git`
- Switch to dev branch: `git checkout dev`

---

## 2. Solution Structure & MVVM Architecture

```
ImageClusterizer/
  ImageClusterizer.sln
  ImageClusterizer/               <- Main WPF Application
    App.xaml
    MainWindow.xaml               <- Shell window
    Models/                       <- Data entities
      ImageRecord.cs              <- Image path + embedding vector
      ClusterResult.cs            <- Cluster ID + image list
    ViewModels/                   <- Application logic (MVVM)
      MainViewModel.cs            <- Main orchestrator VM
      ScanViewModel.cs            <- Folder scan logic
      ClusterViewModel.cs         <- Clustering result VM
    Views/                        <- XAML views (no code-behind logic)
      MainView.xaml
      ScanView.xaml
      ClusterView.xaml
    Services/                     <- Business services
      OnnxService.cs              <- ResNet-50 inference
      VectorService.cs            <- Cosine + PCA computation
      CacheService.cs             <- SQLite embedding cache
    Helpers/
      ImagePreprocessor.cs        <- Normalize + resize to 224x224
    Models/ONNX/
      resnet50-v2-7.onnx          <- Model file (not in git, download separately)
  Polygon/                        <- Research sandbox
    Polygon1_BasicInference/
    Polygon2_CosineSimilarity/
    Polygon3_PCA_RSVD/
    Polygon4_SQLiteCache/
```

### MVVM Rules
- **Views** contain ZERO business logic — only XAML bindings
- **ViewModels** expose `ObservableProperty` / `RelayCommand` (CommunityToolkit.Mvvm)
- **Models** are plain data containers (no UI references)
- **Services** are injected via constructor or property injection
- No `System.Windows` references in ViewModels or Models

---

## 3. NuGet Packages

```xml
<!-- ImageClusterizer.csproj -->
<ItemGroup>
  <!-- ONNX Runtime — CPU inference -->
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.17.3" />

  <!-- MVVM toolkit — RelayCommand, ObservableProperty, etc. -->
  <PackageReference Include="CommunityToolkit.Mvvm" Version="8.2.2" />

  <!-- SQLite — local embedding cache (no server needed) -->
  <PackageReference Include="Microsoft.Data.Sqlite" Version="8.0.0" />

  <!-- Image loading (BitmapImage helper) -->
  <PackageReference Include="System.Drawing.Common" Version="8.0.0" />
</ItemGroup>
```

**Why ONNX Runtime instead of ML.NET full pipeline?**
ONNX Runtime loads the `.onnx` model file directly — no retraining, no Python bridge.
It runs purely on CPU by default, keeping the app self-contained.
For GPU acceleration add `Microsoft.ML.OnnxRuntime.Gpu` and CUDA 12.x drivers.
---

## 4. ResNet-50 & ONNX Integration

### 4.1 Download the ONNX Model

```bash
# From ONNX Model Zoo (official)
# https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
# File: resnet50-v2-7.onnx (~97 MB)
# Place in: ImageClusterizer/Models/ONNX/resnet50-v2-7.onnx
```

### 4.2 OnnxService — Session Lifecycle

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class OnnxService : IDisposable
{
    // Input name for ResNet-50 v2 ONNX model
    private const string InputName  = "data";
    // We tap the pre-softmax layer (2048-dim) for embeddings
    // ResNet-50 v2 output node: "resnetv24_dense0_fwd" (1000-dim softmax)
    // For 2048-dim embeddings use the penultimate layer output
    // Simplest approach: use the 1000-dim logit vector as embedding
    private const string OutputName = "resnetv24_dense0_fwd";

    private readonly InferenceSession _session;

    public OnnxService(string modelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        // Enable intra-op parallelism (use all CPU cores for a single inference)
        opts.IntraOpNumThreads = Environment.ProcessorCount;
        _session = new InferenceSession(modelPath, opts);
    }

    // Returns float[1000] — logit vector (pre-softmax activations)
    public float[] GetEmbedding(float[] preprocessedPixels)
    {
        // Shape: [batch=1, channels=3, height=224, width=224]
        var tensor = new DenseTensor<float>(preprocessedPixels,
            new[] { 1, 3, 224, 224 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, tensor)
        };

        using var results = _session.Run(inputs);
        return results[0].AsEnumerable<float>().ToArray();
    }

    public void Dispose() => _session.Dispose();
}
```

### 4.3 ONNX Session Options — Performance

| Option | Value | Effect |
|--------|-------|--------|
| `GraphOptimizationLevel` | `ORT_ENABLE_ALL` | Fuses ops, prunes dead nodes |
| `IntraOpNumThreads` | `Environment.ProcessorCount` | Parallel matrix ops within one inference |
| `ExecutionMode` | `ORT_SEQUENTIAL` | Best for single-image inference |
| `InterOpNumThreads` | 1 | Irrelevant for sequential mode |
---

## 5. Image Preprocessing Pipeline

ResNet-50 expects: `float32[1, 3, 224, 224]` in CHW format, normalized with ImageNet stats.

```csharp
using System.Drawing;
using System.Drawing.Imaging;

public static class ImagePreprocessor
{
    // ImageNet channel mean & std (RGB order)
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

    public static float[] Preprocess(string imagePath)
    {
        using var original = new Bitmap(imagePath);
        using var resized  = new Bitmap(original, new Size(224, 224));

        var pixels = new float[3 * 224 * 224]; // CHW layout

        // Lock bits for fast pixel access (avoid GetPixel() — too slow)
        var bmpData = resized.LockBits(
            new Rectangle(0, 0, 224, 224),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        unsafe
        {
            byte* ptr = (byte*)bmpData.Scan0.ToPointer();
            int stride = bmpData.Stride;

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    // Format24bppRgb is stored as B, G, R
                    byte b = ptr[y * stride + x * 3 + 0];
                    byte g = ptr[y * stride + x * 3 + 1];
                    byte r = ptr[y * stride + x * 3 + 2];

                    float rf = (r / 255f - Mean[0]) / Std[0];
                    float gf = (g / 255f - Mean[1]) / Std[1];
                    float bf = (b / 255f - Mean[2]) / Std[2];

                    int idx = y * 224 + x;
                    pixels[0 * 224 * 224 + idx] = rf; // R channel
                    pixels[1 * 224 * 224 + idx] = gf; // G channel
                    pixels[2 * 224 * 224 + idx] = bf; // B channel
                }
            }
        }
        resized.UnlockBits(bmpData);
        return pixels;
    }
}
```

**Key points:**
- `LockBits` + unsafe pointer = ~20x faster than `GetPixel()`
- Format24bppRgb stores bytes as B, G, R (not RGB) — order matters
- Normalize each channel: `(value/255 - mean) / std`
- Output layout is CHW (channel-first), not HWC
---

## 6. Embedding Vectors — Extraction & Storage

### 6.1 What is an Embedding Vector?

After inference, ResNet-50 produces a `float[1000]` vector — the **logit vector** (pre-softmax activations).
This vector encodes the visual fingerprint of the image in 1000 dimensions.
Images that look similar will have vectors that point in similar directions (high cosine similarity).

### 6.2 Serialization to Byte Array for SQLite

```csharp
// Convert float[] to byte[] for DB storage
public static byte[] ToBytes(float[] vector)
{
    var bytes = new byte[vector.Length * sizeof(float)];
    Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
    return bytes;
}

// Convert byte[] back to float[]
public static float[] FromBytes(byte[] data)
{
    var vector = new float[data.Length / sizeof(float)];
    Buffer.BlockCopy(data, 0, vector, 0, data.Length);
    return vector;
}
```

### 6.3 L2 Normalization (recommended before cosine/PCA)

```csharp
public static float[] L2Normalize(float[] v)
{
    float norm = MathF.Sqrt(v.Sum(x => x * x));
    if (norm < 1e-10f) return v;
    return v.Select(x => x / norm).ToArray();
}
```

After L2 normalization, cosine similarity = dot product (no division needed).

---

## 7. Cosine Similarity — Fast Implementation

### 7.1 Naive (baseline)

```csharp
public static float CosineSimilarityNaive(float[] a, float[] b)
{
    float dot = 0f, normA = 0f, normB = 0f;
    for (int i = 0; i < a.Length; i++)
    {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-10f);
}
```

### 7.2 SIMD-Accelerated (System.Numerics.Vector)

```csharp
using System.Numerics;

public static float CosineSimilaritySimd(float[] a, float[] b)
{
    int n = a.Length;
    int step = Vector<float>.Count; // typically 8 on AVX2

    var vDot   = Vector<float>.Zero;
    var vNormA = Vector<float>.Zero;
    var vNormB = Vector<float>.Zero;

    int i = 0;
    for (; i <= n - step; i += step)
    {
        var va = new Vector<float>(a, i);
        var vb = new Vector<float>(b, i);
        vDot   += va * vb;
        vNormA += va * va;
        vNormB += vb * vb;
    }

    float dot   = Vector.Dot(vDot,   Vector<float>.One);
    float normA = Vector.Dot(vNormA, Vector<float>.One);
    float normB = Vector.Dot(vNormB, Vector<float>.One);

    // Handle remaining elements (tail)
    for (; i < n; i++)
    {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dot / (MathF.Sqrt(normA * normB) + 1e-10f);
}
```

### 7.3 If Vectors are Pre-Normalized (L2 = 1.0)

```csharp
// Fastest: just a dot product
public static float CosineSimilarityNormalized(float[] a, float[] b)
{
    var va = new ReadOnlySpan<float>(a);
    var vb = new ReadOnlySpan<float>(b);
    float dot = 0f;
    for (int i = 0; i < va.Length; i++)
        dot += va[i] * vb[i];
    return dot; // already in [-1, 1]
}
```

### 7.4 All-Pairs Matrix Computation (N images)

```csharp
// Returns N x N similarity matrix
// Uses Parallel.For — exploits all CPU cores
public static float[,] BuildSimilarityMatrix(float[][] embeddings)
{
    int n = embeddings.Length;
    var matrix = new float[n, n];

    Parallel.For(0, n, i =>
    {
        for (int j = i; j < n; j++)
        {
            float s = CosineSimilaritySimd(embeddings[i], embeddings[j]);
            matrix[i, j] = s;
            matrix[j, i] = s; // symmetric
        }
    });

    return matrix;
}
```

**Performance tips for cosine:**
- Pre-normalize all vectors once → reduce per-pair division to zero
- Use `Parallel.For` for the outer loop over N images
- SIMD gives ~4–8x speedup vs scalar on AVX2 CPUs
- For 10,000 images → 50M pairs → ~2s on modern 8-core CPU with SIMD
---

## 8. PCA via Randomized SVD (RSVD/Halko)

### 8.1 Why PCA?

1000-dimensional vectors are expensive for distance computation and clustering.
PCA reduces them to e.g. 50 or 128 dimensions while preserving most variance.

### 8.2 Why Randomized SVD instead of Classic SVD?

| Method | Time Complexity | Good For |
|--------|----------------|----------|
| Classic SVD (LAPACK) | O(n * d^2) | Small datasets |
| Randomized SVD (Halko 2011) | O(n * d * k) | Large datasets, k << d |

For 1000-dim vectors and k=50 components: RSVD is ~10x faster than full SVD.

### 8.3 RSVD Implementation (no external math library needed)

```csharp
public static class RandomizedPCA
{
    // Input:  X[n, d] — n images, d-dimensional embeddings
    // Output: reduced[n, k] — n images in k-dimensional PCA space
    public static float[,] FitTransform(float[,] X, int k = 50, int powerIter = 2)
    {
        int n = X.GetLength(0);
        int d = X.GetLength(1);

        // Step 1: Center the data (subtract column means)
        float[] means = ColumnMeans(X, n, d);
        float[,] Xc = Center(X, means, n, d);

        // Step 2: Random projection matrix Omega [d, k+oversampling]
        int l = k + 10; // oversampling for accuracy
        var rng = new Random(42);
        float[,] Omega = RandomGaussian(d, l, rng);

        // Step 3: Y = Xc * Omega  [n, l]
        float[,] Y = Multiply(Xc, Omega, n, d, l);

        // Step 4: Power iterations for accuracy: Y = (Xc * Xc^T)^q * Y
        for (int q = 0; q < powerIter; q++)
        {
            // Y = Xc * (Xc^T * Y)
            float[,] Z = MultiplyTransposed(Xc, Y, n, d, l);  // [d, l]
            Y = Multiply(Xc, Z, n, d, l);                      // [n, l]
        }

        // Step 5: QR decomposition of Y to get orthonormal basis Q [n, l]
        float[,] Q = QRDecomposition(Y, n, l);

        // Step 6: B = Q^T * Xc  [l, d]
        float[,] B = MultiplyQtX(Q, Xc, n, d, l);

        // Step 7: SVD of small matrix B [l, d] -> take top k singular vectors
        (float[,] Vt, float[] S) = SmallSVD(B, l, d, k);

        // Step 8: Project original data: result = Xc * V  [n, k]
        float[,] V = Transpose(Vt, k, d);
        return Multiply(Xc, V, n, d, k);
    }

    // --- Helper: Column means ---
    private static float[] ColumnMeans(float[,] X, int n, int d)
    {
        var means = new float[d];
        for (int j = 0; j < d; j++)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += X[i, j];
            means[j] = sum / n;
        }
        return means;
    }

    // --- Helper: Center matrix ---
    private static float[,] Center(float[,] X, float[] means, int n, int d)
    {
        var C = new float[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                C[i, j] = X[i, j] - means[j];
        return C;
    }

    // --- Helper: Random Gaussian matrix ---
    private static float[,] RandomGaussian(int rows, int cols, Random rng)
    {
        var M = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                M[i, j] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) *
                                  Math.Cos(2.0 * Math.PI * u2));
            }
        return M;
    }

    // --- Helper: Matrix multiply A[n,d] * B[d,l] -> C[n,l] ---
    private static float[,] Multiply(float[,] A, float[,] B, int n, int d, int l)
    {
        var C = new float[n, l];
        Parallel.For(0, n, i =>
        {
            for (int k = 0; k < d; k++)
            {
                float aik = A[i, k];
                for (int j = 0; j < l; j++)
                    C[i, j] += aik * B[k, j];
            }
        });
        return C;
    }

    // (other helpers: MultiplyTransposed, QRDecomposition, SmallSVD, Transpose)
    // See Polygon3_PCA_RSVD for full reference implementation
}
```

### 8.4 QR Decomposition (Gram-Schmidt)

```csharp
// Orthonormalizes columns of Y using Gram-Schmidt process
private static float[,] QRDecomposition(float[,] Y, int n, int l)
{
    var Q = new float[n, l];
    for (int j = 0; j < l; j++)
    {
        // Copy column j
        var col = new float[n];
        for (int i = 0; i < n; i++) col[i] = Y[i, j];

        // Subtract projections onto previous columns
        for (int k = 0; k < j; k++)
        {
            float dot = 0f;
            for (int i = 0; i < n; i++) dot += Q[i, k] * col[i];
            for (int i = 0; i < n; i++) col[i] -= dot * Q[i, k];
        }

        // Normalize
        float norm = MathF.Sqrt(col.Sum(x => x * x));
        if (norm > 1e-10f)
            for (int i = 0; i < n; i++) Q[i, j] = col[i] / norm;
    }
    return Q;
}
```

### 8.5 Caching PCA Coordinates in SQLite

```csharp
// Store reduced vectors to avoid recomputing PCA on every launch
public void SavePCACoordinates(string imagePath, float[] coords)
{
    using var cmd = _conn.CreateCommand();
    cmd.CommandText = @"INSERT OR REPLACE INTO pca_cache
        (path, coords, updated_at) VALUES ($p, $c, $t)";
    cmd.Parameters.AddWithValue("$p", imagePath);
    cmd.Parameters.AddWithValue("$c", VectorHelper.ToBytes(coords));
    cmd.Parameters.AddWithValue("$t", DateTime.UtcNow.Ticks);
    cmd.ExecuteNonQuery();
}
```
---

## 9. SQLite Caching Layer

### 9.1 Why Cache Embeddings?

ResNet-50 inference takes ~50–200ms per image.
For a folder of 1000 images the first scan takes minutes.
SQLite caches vectors by image path + file hash → subsequent scans are instant.

### 9.2 CacheService Full Pattern

```csharp
using Microsoft.Data.Sqlite;

public class CacheService : IDisposable
{
    private readonly SqliteConnection _conn;

    public CacheService(string dbPath = "embeddings.db")
    {
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        InitSchema();
    }

    private void InitSchema()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS embeddings (
                path       TEXT PRIMARY KEY,
                file_hash  TEXT NOT NULL,
                vector     BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pca_cache (
                path       TEXT PRIMARY KEY,
                coords     BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );
        ";
        cmd.ExecuteNonQuery();
    }

    // Get cached embedding, or null if not found / hash changed
    public float[]? GetEmbedding(string path)
    {
        string currentHash = ComputeHash(path);
        using var cmd = _conn.CreateCommand();
        cmd.CommandText =
            "SELECT vector FROM embeddings WHERE path=$p AND file_hash=$h";
        cmd.Parameters.AddWithValue("$p", path);
        cmd.Parameters.AddWithValue("$h", currentHash);

        var result = cmd.ExecuteScalar();
        if (result is byte[] bytes)
            return VectorHelper.FromBytes(bytes);
        return null;
    }

    // Save embedding to cache
    public void SaveEmbedding(string path, float[] vector)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = @"INSERT OR REPLACE INTO embeddings
            (path, file_hash, vector, created_at) VALUES ($p, $h, $v, $t)";
        cmd.Parameters.AddWithValue("$p", path);
        cmd.Parameters.AddWithValue("$h", ComputeHash(path));
        cmd.Parameters.AddWithValue("$v", VectorHelper.ToBytes(vector));
        cmd.Parameters.AddWithValue("$t", DateTime.UtcNow.Ticks);
        cmd.ExecuteNonQuery();
    }

    // MD5 hash of file contents — detects if image was modified
    private static string ComputeHash(string path)
    {
        using var md5 = System.Security.Cryptography.MD5.Create();
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(md5.ComputeHash(stream));
    }

    public void Dispose() => _conn.Dispose();
}
```

### 9.3 Batch Insert with Transaction

```csharp
// For initial scan of 1000+ images: wrap in transaction for ~50x speedup
public void SaveEmbeddingsBatch(IEnumerable<(string path, float[] vector)> items)
{
    using var transaction = _conn.BeginTransaction();
    using var cmd = _conn.CreateCommand();
    cmd.Transaction = transaction;
    cmd.CommandText = @"INSERT OR REPLACE INTO embeddings
        (path, file_hash, vector, created_at) VALUES ($p, $h, $v, $t)";

    var pParam = cmd.Parameters.Add("$p", SqliteType.Text);
    var hParam = cmd.Parameters.Add("$h", SqliteType.Text);
    var vParam = cmd.Parameters.Add("$v", SqliteType.Blob);
    var tParam = cmd.Parameters.Add("$t", SqliteType.Integer);

    foreach (var (path, vector) in items)
    {
        pParam.Value = path;
        hParam.Value = ComputeHash(path);
        vParam.Value = VectorHelper.ToBytes(vector);
        tParam.Value = DateTime.UtcNow.Ticks;
        cmd.ExecuteNonQuery();
    }

    transaction.Commit();
}
```
---

## 10. Clustering Pipeline

### 10.1 Full Pipeline Flow

```
[Folder path]
    |
    v
[ScanViewModel] -- finds all .jpg/.png/.bmp files
    |
    v
[CacheService.GetEmbedding()] -- check SQLite cache
    |   cached: return float[] immediately
    |   not cached:
    v
[ImagePreprocessor.Preprocess()] -- resize 224x224, normalize
    |
    v
[OnnxService.GetEmbedding()] -- ResNet-50 inference -> float[1000]
    |
    v
[CacheService.SaveEmbedding()] -- persist to SQLite
    |
    v
[VectorService.L2Normalize()] -- all vectors -> unit sphere
    |
    v
[RandomizedPCA.FitTransform()] -- 1000-dim -> 50-dim
    |
    v
[Cluster()] -- group by cosine similarity threshold
    |
    v
[ClusterViewModel] -- display results in WPF DataGrid/ListView
```

### 10.2 Simple Threshold Clustering

```csharp
// Groups images where pairwise cosine similarity >= threshold
public static List<List<int>> ClusterByThreshold(float[][] embeddings,
    float threshold = 0.85f)
{
    int n = embeddings.Length;
    var visited = new bool[n];
    var clusters = new List<List<int>>();

    for (int i = 0; i < n; i++)
    {
        if (visited[i]) continue;
        var cluster = new List<int> { i };
        visited[i] = true;

        for (int j = i + 1; j < n; j++)
        {
            if (visited[j]) continue;
            float sim = CosineSimilaritySimd(embeddings[i], embeddings[j]);
            if (sim >= threshold)
            {
                cluster.Add(j);
                visited[j] = true;
            }
        }

        clusters.Add(cluster);
    }

    return clusters;
}
```

---

## 11. WPF UI Patterns

### 11.1 CommunityToolkit.Mvvm — ViewModel Basics

```csharp
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private string _status = "Ready";

    [ObservableProperty]
    private int _progressValue;

    [ObservableProperty]
    private ObservableCollection<ClusterViewModel> _clusters = new();

    [RelayCommand(CanExecute = nameof(CanScan))]
    private async Task ScanAsync()
    {
        Status = "Scanning...";
        // ... call services
    }

    private bool CanScan() => !string.IsNullOrEmpty(FolderPath);
}
```

### 11.2 Progress Reporting (IProgress<T>)

```csharp
// In ViewModel (UI thread safe)
var progress = new Progress<(int current, int total, string file)>(p =>
{
    ProgressValue = (int)(p.current * 100.0 / p.total);
    Status = $"Processing {p.current}/{p.total}: {Path.GetFileName(p.file)}";
});

// In Service (background thread)
await Task.Run(() =>
{
    for (int i = 0; i < files.Count; i++)
    {
        ProcessFile(files[i]);
        ((IProgress<(int,int,string)>)progress).Report((i+1, files.Count, files[i]));
    }
});
```

### 11.3 Image Loading with Thumbnail Cache

```csharp
// Load image for display without locking the file
public static BitmapImage LoadThumbnail(string path, int maxSize = 200)
{
    var bitmap = new BitmapImage();
    bitmap.BeginInit();
    bitmap.UriSource = new Uri(path);
    bitmap.DecodePixelWidth = maxSize; // decode at thumbnail size = less memory
    bitmap.CacheOption = BitmapCacheOption.OnLoad; // release file lock immediately
    bitmap.EndInit();
    bitmap.Freeze(); // make thread-safe for UI binding
    return bitmap;
}
```
---

## 12. Polygon — Research Sandbox

The `Polygon/` folder contains self-contained experiments.
Each Polygon is a standalone .NET console project demonstrating one concept.

| Folder | What it demonstrates |
|--------|---------------------|
| `Polygon1_BasicInference/` | Load ONNX model, run on one image, print top-5 classes |
| `Polygon2_CosineSimilarity/` | Compare two images, print similarity score + SIMD benchmark |
| `Polygon3_PCA_RSVD/` | Run RSVD on 500 random vectors, compare with full SVD timing |
| `Polygon4_SQLiteCache/` | Demonstrate cache hit/miss logic, batch insert benchmark |
| `Polygon5_Clustering/` | Run threshold clustering on 100 images, visualize cluster sizes |

### How to create a new Polygon

```bash
cd Polygon
mkdir Polygon6_MyExperiment
cd Polygon6_MyExperiment
dotnet new console -n Polygon6_MyExperiment
dotnet add package Microsoft.ML.OnnxRuntime
```

Each Polygon has its own `README.md` explaining the experiment and expected output.

---

## 13. Performance Checklist

### Image Processing
- [ ] Use `LockBits` + unsafe pointer, never `GetPixel()`
- [ ] Process images in parallel: `Parallel.ForEach(files, ProcessFile)`
- [ ] Cache embeddings in SQLite by file hash
- [ ] Skip unchanged files (hash unchanged = use cached vector)

### Vector Computation
- [ ] L2-normalize all vectors once after extraction
- [ ] Use SIMD (`System.Numerics.Vector<float>`) for cosine computation
- [ ] Use `Parallel.For` for N×N similarity matrix construction
- [ ] Apply RSVD (k=50) before building full similarity matrix for large datasets
- [ ] Use power iterations (powerIter=2) in RSVD for better accuracy

### SQLite
- [ ] Use `BeginTransaction()` / `Commit()` for batch inserts
- [ ] Use `INSERT OR REPLACE` (upsert) pattern
- [ ] Add index on `path` column (already PRIMARY KEY = indexed)
- [ ] Enable WAL mode: `PRAGMA journal_mode=WAL` for concurrent reads

### WPF / UI
- [ ] Run all heavy processing on background threads (`Task.Run`)
- [ ] Use `IProgress<T>` for thread-safe UI updates
- [ ] `bitmap.Freeze()` before passing to UI thread
- [ ] Use `DecodePixelWidth` for thumbnail loading (reduce memory 10x)
- [ ] Virtualize lists: use `VirtualizingStackPanel` in ListViews

### ONNX
- [ ] Create `InferenceSession` once at startup (not per image)
- [ ] Enable `ORT_ENABLE_ALL` graph optimization
- [ ] Set `IntraOpNumThreads = ProcessorCount`
- [ ] Consider GPU (OnnxRuntime.Gpu) for >10,000 images

---

## 14. Build & Run from Zero

```bash
# 1. Clone repository
git clone https://github.com/KirinDenis/ImageClusterizer.git
cd ImageClusterizer
git checkout dev

# 2. Download ONNX model (resnet50-v2-7.onnx, ~97 MB)
# From: https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
# Place at: ImageClusterizer/Models/ONNX/resnet50-v2-7.onnx

# 3. Restore & build
dotnet restore
dotnet build --configuration Release

# 4. Run (WPF requires Windows)
dotnet run --project ImageClusterizer

# Or open in Visual Studio 2022:
# File -> Open -> Solution -> ImageClusterizer.sln
# Press F5
```

### First Launch Checklist
1. Select a folder containing images (JPG/PNG/BMP)
2. First scan: embeddings computed and cached to `embeddings.db`
3. Subsequent scans: instant (vectors loaded from SQLite)
4. Adjust similarity threshold slider to control cluster granularity
5. Explore Polygon projects for deep dives on individual algorithms

---

## References

- **Halko N., Martinsson P., Tropp J. (2011)** — *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions* — the mathematical foundation for RSVD
- **ONNX Model Zoo** — https://github.com/onnx/models
- **Microsoft.ML.OnnxRuntime docs** — https://onnxruntime.ai/docs/get-started/with-csharp.html
- **CommunityToolkit.Mvvm** — https://learn.microsoft.com/en-us/dotnet/communitytoolkit/mvvm/
- **ImageClusterizer Wiki** — https://github.com/KirinDenis/ImageClusterizer/wiki
- **Polygon research projects** — see `/Polygon` folder in this branch
