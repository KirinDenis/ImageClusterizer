using ImageClusterizer.Models;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

// Alias to resolve ambiguity between MathNet.Numerics.LinearAlgebra.Vector<T> and System.Numerics.Vector<T>
// System.Numerics.Vector<T> provides SIMD (Single Instruction Multiple Data) operations
// Reference: https://docs.microsoft.com/en-us/dotnet/api/system.numerics.vector-1
using SysVector = System.Numerics.Vector;
using SysVectorT = System.Numerics.Vector<float>;

namespace ImageClusterizer.Services;

/// <summary>
/// High-performance clustering service using SIMD optimizations and parallel processing.
/// Implements similarity-based clustering using cosine similarity metric.
/// 
/// References:
/// - Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
/// - PCA Dimensionality Reduction: https://en.wikipedia.org/wiki/Principal_component_analysis
/// - SIMD in .NET: https://docs.microsoft.com/en-us/dotnet/standard/simd
/// - AVX2 Instructions: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
/// </summary>
public class ClusteringService
{
    // ArrayPool reduces GC pressure by reusing array instances
    // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.buffers.arraypool-1
    private static readonly ArrayPool<float> FloatPool = ArrayPool<float>.Shared;
    private static readonly ArrayPool<double> DoublePool = ArrayPool<double>.Shared;

    /// <summary>
    /// Clusters image vectors by cosine similarity using parallel processing.
    /// Uses a greedy algorithm where each unassigned vector becomes a cluster seed,
    /// and similar vectors (above threshold) are grouped together.
    /// 
    /// Algorithm Complexity: O(n²) for similarity comparisons, parallelized across CPU cores
    /// 
    /// References:
    /// - Parallel.ForEach: https://docs.microsoft.com/en-us/dotnet/api/system.threading.tasks.parallel.foreach
    /// - ConcurrentDictionary: https://docs.microsoft.com/en-us/dotnet/api/system.collections.concurrent.concurrentdictionary-2
    /// </summary>
    /// <param name="vectors">Collection of image feature vectors to cluster</param>
    /// <param name="similarityThreshold">Minimum cosine similarity (0-1) to group vectors together. Default: 0.85</param>
    /// <returns>List of clusters, each containing similar images and their centroid</returns>
    public List<ImageCluster> ClusterBySimilarity(
        List<ImageVector> vectors,
        float similarityThreshold = 0.85f)
    {
        var clusters = new List<ImageCluster>();

        // Thread-safe dictionary to track which images have been assigned to clusters
        // TryAdd provides atomic check-and-set operation
        var assigned = new ConcurrentDictionary<string, bool>();

        // Lock object for synchronized access to the clusters list
        var clustersLock = new object();

        // Parallel processing: each thread processes different seed vectors simultaneously
        // MaxDegreeOfParallelism set to CPU core count for optimal performance
        // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.threading.tasks.paralleloptions
        Parallel.ForEach(vectors, new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        },
        vector =>
        {
            // Atomically check if this vector is already assigned to a cluster
            // TryAdd returns false if key already exists (thread-safe operation)
            if (!assigned.TryAdd(vector.FilePath, true))
                return;

            // Create new cluster with this vector as the seed
            var cluster = new ImageCluster
            {
                Images = new List<ImageVector> { vector }
            };

            // Thread-safe collection for accumulating similar images
            // ConcurrentBag is optimized for scenarios where same thread does most adds/takes
            // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.collections.concurrent.concurrentbag-1
            var similarImages = new ConcurrentBag<ImageVector>();

            // Nested parallel loop to compare seed vector with all candidates
            // Each thread compares different candidate vectors simultaneously
            Parallel.ForEach(vectors, candidate =>
            {
                // Skip if candidate already assigned to another cluster
                if (assigned.ContainsKey(candidate.FilePath))
                    return;

                // Create ReadOnlySpan for zero-copy, bounds-check-free access to vector data
                // Span<T> is a ref struct that provides safe, efficient access to contiguous memory
                // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.span-1
                ReadOnlySpan<float> vectorSpan = vector.Vector;
                ReadOnlySpan<float> candidateSpan = candidate.Vector;

                // Compute cosine similarity using SIMD-optimized implementation
                // SIMD allows processing multiple float values in single CPU instruction
                var similarity = CosineSimilaritySIMD(vectorSpan, candidateSpan);

                // If similarity exceeds threshold, attempt to add to cluster
                if (similarity >= similarityThreshold)
                {
                    // Atomic check-and-add: ensures image isn't added to multiple clusters
                    if (assigned.TryAdd(candidate.FilePath, true))
                    {
                        similarImages.Add(candidate);
                    }
                }
            });

            // Merge similar images into the cluster
            cluster.Images.AddRange(similarImages);

            // Calculate cluster centroid (mean vector) using parallel SIMD operations
            cluster.Centroid = CalculateCentroidOptimized(cluster.Images);

            // Thread-safe addition of cluster to results
            // Lock ensures clusters list isn't corrupted by concurrent modifications
            lock (clustersLock)
            {
                cluster.ClusterId = clusters.Count;
                clusters.Add(cluster);
            }
        });

        return clusters;
    }

    /// <summary>
    /// Dispatches cosine similarity calculation to optimal SIMD implementation based on:
    /// 1. Vector length (small vectors use scalar code to avoid SIMD overhead)
    /// 2. CPU capabilities (AVX2 > Vector&lt;T&gt; > Scalar)
    /// 
    /// AggressiveInlining hint requests compiler to inline this method for zero-cost abstraction
    /// Reference: https://docs.microsoft.com/en-us/dotnet/api/system.runtime.compilerservices.methodimploptions
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float CosineSimilaritySIMD(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        int length = a.Length;

        // For small vectors, SIMD overhead exceeds benefits - use scalar code
        // Threshold: 4x vector width (e.g., 4*8=32 for AVX2)
        if (length < SysVectorT.Count * 4)
        {
            return CosineSimilarityScalar(a, b);
        }

        // Check if CPU supports AVX2 instructions (8 floats processed per instruction)
        // AVX2 introduced in Intel Haswell (2013) and AMD Excavator (2015)
        // Reference: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
        if (Avx2.IsSupported && length >= 8)
        {
            return CosineSimilarityAVX2(a, b);
        }

        // Fallback to System.Numerics.Vector<T> - cross-platform SIMD
        // Automatically uses best available instruction set (SSE, AVX, NEON on ARM)
        return CosineSimilarityVector(a, b);
    }

    /// <summary>
    /// Scalar (non-SIMD) implementation of cosine similarity.
    /// Used as fallback for small vectors or when SIMD unavailable.
    /// 
    /// Formula: cos(θ) = (A·B) / (||A|| * ||B||)
    /// where A·B is dot product, ||A|| is magnitude of vector A
    /// 
    /// Reference: https://en.wikipedia.org/wiki/Cosine_similarity
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float CosineSimilarityScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dotProduct = 0f;    // A·B = Σ(a[i] * b[i])
        float magnitudeA = 0f;    // ||A|| = √(Σ(a[i]²))
        float magnitudeB = 0f;    // ||B|| = √(Σ(b[i]²))

        // Single loop calculates all three values simultaneously
        for (int i = 0; i < a.Length; i++)
        {
            float va = a[i];
            float vb = b[i];

            dotProduct += va * vb;
            magnitudeA += va * va;
            magnitudeB += vb * vb;
        }

        // Return cosine similarity: dot product divided by product of magnitudes
        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
    }

    /// <summary>
    /// AVX2-optimized cosine similarity using 256-bit vector operations (8 floats at once).
    /// 
    /// Performance: ~8x faster than scalar code for large vectors (2048 dimensions)
    /// 
    /// AggressiveOptimization enables maximum compiler optimizations for this method
    /// 'unsafe' keyword required for pointer arithmetic with fixed statement
    /// 
    /// References:
    /// - AVX2 Intrinsics: https://docs.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics.x86.avx2
    /// - Unsafe Code: https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/unsafe-code
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float CosineSimilarityAVX2(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = a.Length;
        int vectorSize = Vector256<float>.Count; // 8 floats per 256-bit vector
        int vectorCount = length / vectorSize;   // Number of full vectors we can process

        // Initialize 256-bit accumulators for SIMD calculations
        // Vector256<float> represents 8 floats that can be processed in parallel
        Vector256<float> dotProductVec = Vector256<float>.Zero;
        Vector256<float> magnitudeAVec = Vector256<float>.Zero;
        Vector256<float> magnitudeBVec = Vector256<float>.Zero;

        // Pin arrays in memory to get direct pointer access
        // 'fixed' prevents garbage collector from moving arrays during pointer operations
        // Reference: https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/statements/fixed
        fixed (float* ptrA = a)
        fixed (float* ptrB = b)
        {
            // Process 8 floats per iteration using AVX2 vector instructions
            for (int i = 0; i < vectorCount; i++)
            {
                int offset = i * vectorSize;

                // Load 8 consecutive floats from memory into 256-bit registers
                // Avx.LoadVector256 is single CPU instruction (VMOVUPS)
                var vecA = Avx.LoadVector256(ptrA + offset);
                var vecB = Avx.LoadVector256(ptrB + offset);

                // Parallel multiply-add operations on 8 floats simultaneously
                // Each operation is single CPU instruction (VFMADD, VMULPS, VADDPS)
                dotProductVec = Avx.Add(dotProductVec, Avx.Multiply(vecA, vecB));
                magnitudeAVec = Avx.Add(magnitudeAVec, Avx.Multiply(vecA, vecA));
                magnitudeBVec = Avx.Add(magnitudeBVec, Avx.Multiply(vecB, vecB));
            }
        }

        // Reduce 8-element vectors to single scalar values by summing all elements
        // HorizontalSum performs tree-reduction using shuffle and add instructions
        float dotProduct = HorizontalSum(dotProductVec);
        float magnitudeA = HorizontalSum(magnitudeAVec);
        float magnitudeB = HorizontalSum(magnitudeBVec);

        // Process remaining elements that don't fit in full vectors (tail processing)
        // Example: for length=2050 and vectorSize=8, processes elements 2048-2049
        int remainderStart = vectorCount * vectorSize;
        for (int i = remainderStart; i < length; i++)
        {
            float va = a[i];
            float vb = b[i];

            dotProduct += va * vb;
            magnitudeA += va * va;
            magnitudeB += vb * vb;
        }

        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
    }

    /// <summary>
    /// Horizontal sum reduction: adds all 8 float elements in a Vector256 into single scalar.
    /// 
    /// Uses tree-reduction pattern with shuffle instructions:
    /// [a,b,c,d,e,f,g,h] → [a+e,b+f,c+g,d+h] → [a+c+e+g,b+d+f+h] → [a+b+c+d+e+f+g+h]
    /// 
    /// Performance: 4 instructions vs 7 additions for naive loop
    /// 
    /// References:
    /// - Horizontal Operations: https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
    /// - SSE Shuffle: https://docs.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics.x86.sse.shuffle
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float HorizontalSum(Vector256<float> vector)
    {
        // Split 256-bit vector into two 128-bit halves
        // ExtractVector128 is zero-cost operation (just register renaming)
        var lower = Avx.ExtractVector128(vector, 0);  // [a,b,c,d]
        var upper = Avx.ExtractVector128(vector, 1);  // [e,f,g,h]
        var sum128 = Sse.Add(lower, upper);            // [a+e, b+f, c+g, d+h]

        // Shuffle to swap high/low 64-bit pairs
        // 0b_11_10_11_10 = swap elements 2,3 with 0,1
        var shuf = Sse.Shuffle(sum128, sum128, 0b_11_10_11_10);  // [c+g, d+h, a+e, b+f]
        var sum64 = Sse.Add(sum128, shuf);                        // [a+c+e+g, b+d+f+h, *, *]

        // Shuffle to swap 32-bit elements
        shuf = Sse.Shuffle(sum64, sum64, 0b_01_01_01_01);  // [b+d+f+h, *, *, *]
        var sum32 = Sse.Add(sum64, shuf);                   // [a+b+c+d+e+f+g+h, *, *, *]

        // Extract lowest float from 128-bit vector
        return sum32.ToScalar();
    }

    /// <summary>
    /// Cross-platform SIMD implementation using System.Numerics.Vector&lt;T&gt;.
    /// Automatically uses best available instruction set for current CPU:
    /// - x86/x64: SSE2, AVX, AVX2
    /// - ARM: NEON
    /// 
    /// Vector&lt;float&gt;.Count is determined at runtime based on CPU capabilities:
    /// - SSE: 4 floats (128-bit)
    /// - AVX/AVX2: 8 floats (256-bit)
    /// - AVX-512: 16 floats (512-bit)
    /// 
    /// Reference: https://docs.microsoft.com/en-us/dotnet/standard/simd
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float CosineSimilarityVector(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int vectorSize = SysVectorT.Count;  // Runtime-determined based on CPU
        int vectorCount = a.Length / vectorSize;

        // SIMD accumulators - operations performed on all elements simultaneously
        var dotProductVec = SysVectorT.Zero;
        var magnitudeAVec = SysVectorT.Zero;
        var magnitudeBVec = SysVectorT.Zero;

        // Process vectorSize floats per iteration (4-16 depending on CPU)
        for (int i = 0; i < vectorCount; i++)
        {
            int offset = i * vectorSize;

            // Create vectors from span slices - loads data into SIMD registers
            var vecA = new SysVectorT(a.Slice(offset, vectorSize));
            var vecB = new SysVectorT(b.Slice(offset, vectorSize));

            // SIMD operations: += operator is overloaded to use parallel instructions
            dotProductVec += vecA * vecB;  // Multiply-add on all elements
            magnitudeAVec += vecA * vecA;
            magnitudeBVec += vecB * vecB;
        }

        // Reduce SIMD vectors to scalar values using dot product
        // Vector.Dot performs horizontal sum (adds all vector elements)
        // Vector<float>.One is [1,1,1,...,1] vector
        float dotProduct = SysVector.Dot(dotProductVec, SysVectorT.One);
        float magnitudeA = SysVector.Dot(magnitudeAVec, SysVectorT.One);
        float magnitudeB = SysVector.Dot(magnitudeBVec, SysVectorT.One);

        // Process tail elements that don't fill a complete vector
        int remainderStart = vectorCount * vectorSize;
        for (int i = remainderStart; i < a.Length; i++)
        {
            float va = a[i];
            float vb = b[i];

            dotProduct += va * vb;
            magnitudeA += va * va;
            magnitudeB += vb * vb;
        }

        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
    }

    /// <summary>
    /// Calculates cluster centroid (mean vector) using parallel SIMD operations.
    /// Uses thread-local buffers with ArrayPool to minimize allocations and GC pressure.
    /// 
    /// Algorithm: Parallel reduction pattern
    /// 1. Each thread accumulates vectors in local buffer (no synchronization)
    /// 2. Local buffers merged into global result (synchronized)
    /// 3. Divide by count to get mean
    /// 
    /// Performance: O(n*d/p) where n=vectors, d=dimensions, p=cores
    /// 
    /// References:
    /// - Parallel.ForEach with local state: https://docs.microsoft.com/en-us/dotnet/api/system.threading.tasks.parallel.foreach#System_Threading_Tasks_Parallel_ForEach__3
    /// - ArrayPool: https://docs.microsoft.com/en-us/dotnet/api/system.buffers.arraypool-1
    /// </summary>
    private float[] CalculateCentroidOptimized(List<ImageVector> vectors)
    {
        int dimension = vectors[0].Vector.Length;
        int count = vectors.Count;

        // Rent array from pool instead of allocating - reduces GC pressure
        // ArrayPool maintains internal cache of arrays for reuse
        float[] centroid = FloatPool.Rent(dimension);
        Array.Clear(centroid, 0, dimension);  // Zero-initialize rented array

        object lockObj = new object();

        // Three-parameter Parallel.ForEach for thread-local state pattern:
        // 1. localInit: creates thread-local buffer
        // 2. body: accumulates into thread-local buffer (no locks)
        // 3. localFinally: merges thread-local buffer into global result (with lock)
        Parallel.ForEach(vectors,
            // localInit: executed once per thread to create thread-local state
            () =>
            {
                float[] local = FloatPool.Rent(dimension);
                Array.Clear(local, 0, dimension);
                return local;
            },
            // body: executed for each vector, accumulates into thread-local buffer
            // Cannot capture ref struct (Span) in lambda, so we pass arrays
            (vector, loopState, localSum) =>
            {
                // AddVectorsSIMD creates Span internally for SIMD operations
                AddVectorsSIMD(localSum, 0, dimension, vector.Vector, 0, dimension);
                return localSum;
            },
            // localFinally: executed once per thread to merge results
            localSum =>
            {
                // Lock required as multiple threads write to shared 'centroid'
                lock (lockObj)
                {
                    AddVectorsSIMD(centroid, 0, dimension, localSum, 0, dimension);
                }
                // Return buffer to pool for reuse
                FloatPool.Return(localSum);
            });

        // Divide by count using SIMD-optimized division (via multiplication by reciprocal)
        Span<float> centroidSpan = centroid.AsSpan(0, dimension);
        DivideVectorSIMD(centroidSpan, count);

        // Copy result to new array (can't return pooled array to caller)
        float[] result = new float[dimension];
        centroidSpan.CopyTo(result);

        FloatPool.Return(centroid);

        return result;
    }

    /// <summary>
    /// SIMD-optimized vector addition: a += b (element-wise).
    /// Uses System.Numerics.Vector&lt;T&gt; for cross-platform SIMD.
    /// 
    /// Performance: ~4-8x faster than scalar loop for large vectors
    /// 
    /// Note: Takes array parameters (not Span) because Span is ref struct
    /// and cannot be captured in lambda expressions used by Parallel.ForEach
    /// Reference: https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/builtin-types/ref-struct
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AddVectorsSIMD(
        float[] a, int aOffset, int aLength,
        float[] b, int bOffset, int bLength)
    {
        // Create Span views for zero-copy access with bounds checking removed in release builds
        Span<float> spanA = a.AsSpan(aOffset, aLength);
        ReadOnlySpan<float> spanB = b.AsSpan(bOffset, bLength);

        int vectorSize = SysVectorT.Count;
        int vectorCount = aLength / vectorSize;

        // Process multiple elements per iteration using SIMD
        for (int i = 0; i < vectorCount; i++)
        {
            int offset = i * vectorSize;

            var vecA = new SysVectorT(spanA.Slice(offset, vectorSize));
            var vecB = new SysVectorT(spanB.Slice(offset, vectorSize));
            var result = vecA + vecB;  // SIMD addition on all elements

            // Write result back to memory
            result.CopyTo(spanA.Slice(offset, vectorSize));
        }

        // Process remaining elements that don't fill complete vector
        int remainderStart = vectorCount * vectorSize;
        for (int i = remainderStart; i < aLength; i++)
        {
            spanA[i] += spanB[i];
        }
    }

    /// <summary>
    /// SIMD-optimized vector division by scalar using reciprocal multiplication.
    /// 
    /// Optimization: division is ~10x slower than multiplication on modern CPUs.
    /// Instead of: result[i] = vector[i] / divisor (slow)
    /// We compute: result[i] = vector[i] * (1.0f / divisor) (fast)
    /// 
    /// Trade-off: slight precision loss (acceptable for visualization coordinates)
    /// 
    /// Reference: https://en.wikipedia.org/wiki/Division_algorithm#Fast_division_methods
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DivideVectorSIMD(Span<float> vector, int divisor)
    {
        // Compute reciprocal once - reused for all elements
        float multiplier = 1.0f / divisor;
        var multiplierVec = new SysVectorT(multiplier);  // Broadcast to all vector lanes

        int vectorSize = SysVectorT.Count;
        int vectorCount = vector.Length / vectorSize;

        // SIMD multiplication (faster than SIMD division which isn't widely supported)
        for (int i = 0; i < vectorCount; i++)
        {
            int offset = i * vectorSize;

            var vec = new SysVectorT(vector.Slice(offset, vectorSize));
            var result = vec * multiplierVec;  // Parallel multiply on all elements

            result.CopyTo(vector.Slice(offset, vectorSize));
        }

        // Scalar processing for tail elements
        int remainderStart = vectorCount * vectorSize;
        for (int i = remainderStart; i < vector.Length; i++)
        {
            vector[i] *= multiplier;
        }
    }

    /// <summary>
    /// Projects high-dimensional cluster data to 2D coordinates for visualization.
    /// Uses PCA (Principal Component Analysis) for dimensionality reduction,
    /// then normalizes to fit canvas dimensions.
    /// 
    /// Algorithm flow:
    /// 1. Collect all vectors (cluster centroids + individual images)
    /// 2. Reduce from N-dimensions to 2D using PCA
    /// 3. Normalize to canvas size with padding
    /// 
    /// References:
    /// - PCA: https://en.wikipedia.org/wiki/Principal_component_analysis
    /// - Dimensionality Reduction: https://en.wikipedia.org/wiki/Dimensionality_reduction
    /// </summary>
    public List<ClusterPosition> CalculatePositions(
        List<ImageCluster> clusters,
        int canvasWidth = 10000,
        int canvasHeight = 10000)
    {
        var allVectors = new List<float[]>();
        var vectorInfo = new List<VectorInfo>();
        var vectorsLock = new object();

        // Parallel collection of vectors from all clusters
        // Each thread processes different clusters independently
        Parallel.ForEach(clusters, cluster =>
        {
            // Thread-local lists to avoid contention on shared collections
            var localVectors = new List<float[]>();
            var localInfo = new List<VectorInfo>();

            // Add cluster centroid (if exists)
            if (cluster.Centroid != null)
            {
                localVectors.Add(cluster.Centroid);
                localInfo.Add(new VectorInfo
                {
                    ClusterId = cluster.ClusterId,
                    IsCentroid = true,
                    ImageVector = null
                });
            }

            // Add all individual image vectors from this cluster
            foreach (var image in cluster.Images)
            {
                localVectors.Add(image.Vector);
                localInfo.Add(new VectorInfo
                {
                    ClusterId = cluster.ClusterId,
                    IsCentroid = false,
                    ImageVector = image
                });
            }

            // Merge thread-local results into shared collections (synchronized)
            lock (vectorsLock)
            {
                allVectors.AddRange(localVectors);
                vectorInfo.AddRange(localInfo);
            }
        });

        if (allVectors.Count == 0)
            return new List<ClusterPosition>();

        // Reduce dimensionality from N-D to 2D using PCA
        var positions2D = ReduceTo2D_PCA_Optimized(allVectors);

        // Scale and translate to fit canvas with padding
        var normalized = NormalizePositionsOptimized(positions2D, canvasWidth, canvasHeight);

        // Build result array in parallel
        var result = new ClusterPosition[normalized.Length];

        Parallel.For(0, normalized.Length, i =>
        {
            result[i] = new ClusterPosition
            {
                ClusterId = vectorInfo[i].ClusterId,
                IsCentroid = vectorInfo[i].IsCentroid,
                ImageVector = vectorInfo[i].ImageVector,
                X = normalized[i][0],
                Y = normalized[i][1]
            };
        });

        return result.ToList();
    }

    /// <summary>
    /// Performs optimized Principal Component Analysis (PCA) dimensionality reduction.
    /// Reduces high-dimensional vectors (e.g., 2048D from ResNet50) to 2D for visualization.
    /// 
    /// PCA Algorithm:
    /// 1. Center data by subtracting mean of each dimension
    /// 2. Compute covariance matrix
    /// 3. Find eigenvectors via SVD (Singular Value Decomposition)
    /// 4. Project data onto first 2 principal components
    /// 
    /// Uses MathNet.Numerics for SVD computation and custom parallel preprocessing.
    /// 
    /// References:
    /// - PCA Explained: https://en.wikipedia.org/wiki/Principal_component_analysis
    /// - SVD: https://en.wikipedia.org/wiki/Singular_value_decomposition
    /// - MathNet.Numerics: https://numerics.mathdotnet.com/
    /// </summary>
    private double[][] ReduceTo2D_PCA_Optimized(List<float[]> vectors)
    {
        int n = vectors.Count;        // Number of data points
        int d = vectors[0].Length;    // Dimensionality (e.g., 2048)

        // Allocate matrix for all vectors (row-major: each row is one vector)
        var matrixData = new double[n, d];

        // Parallel conversion from float[] to double[,] with SIMD optimizations
        Parallel.For(0, n, i =>
        {
            ReadOnlySpan<float> vectorSpan = vectors[i];

            // MemoryMarshal.CreateSpan provides direct access to 2D array row
            // Avoids bounds checking overhead of [i,j] indexer
            // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.memorymarshal
            Span<double> rowSpan = MemoryMarshal.CreateSpan(ref matrixData[i, 0], d);

            // Convert float to double (required by MathNet)
            for (int j = 0; j < d; j++)
            {
                rowSpan[j] = vectorSpan[j];
            }
        });

        // Create MathNet matrix from array
        // Reference: https://numerics.mathdotnet.com/api/MathNet.Numerics.LinearAlgebra/Matrix%601.htm
        var matrix = Matrix<double>.Build.DenseOfArray(matrixData);

        // Calculate mean of each column (dimension)
        var columnMeans = matrix.ColumnSums() / n;

        // Center data: subtract mean from each element
        // Centering is required for PCA to find principal components
        var centered = matrix.Clone();

        Parallel.For(0, n, i =>
        {
            for (int j = 0; j < d; j++)
            {
                centered[i, j] -= columnMeans[j];
            }
        });

        // Perform Singular Value Decomposition: A = U * Σ * V^T
        // U contains principal components (eigenvectors of AA^T)
        // Σ (s) contains singular values (square roots of eigenvalues)
        // computeVectors=true ensures U matrix is calculated
        // Reference: https://en.wikipedia.org/wiki/Singular_value_decomposition
        var svd = centered.Svd(computeVectors: true);
        var u = svd.U;  // Left singular vectors (n × n)
        var s = svd.S;  // Singular values (min(n,d))

        // Project data onto first 2 principal components
        // Each point: [PC1, PC2] = [u[i,0]*s[0], u[i,1]*s[1]]
        // Scaling by singular values preserves variance information
        var result = new double[n][];

        Parallel.For(0, n, i =>
        {
            result[i] = new double[]
            {
                u[i, 0] * s[0],  // Projection on first principal component
                u[i, 1] * s[1]   // Projection on second principal component
            };
        });

        return result;
    }

    /// <summary>
    /// Normalizes 2D coordinates to fit within canvas bounds with padding.
    /// Applies affine transformation: scale and translate to map data range to canvas range.
    /// 
    /// Optimization: Pre-computes scale and offset to use only multiplication and addition
    /// (avoiding division in inner loop which is ~10x slower than multiplication)
    /// 
    /// Formula: normalized = (original - min) * scale + offset
    /// where: scale = usableSize / range
    ///        offset = padding - min * scale
    /// 
    /// Reference: https://en.wikipedia.org/wiki/Affine_transformation
    /// </summary>
    private double[][] NormalizePositionsOptimized(
        double[][] positions,
        int width,
        int height)
    {
        if (positions.Length == 0) return positions;

        // Find bounding box using PLINQ for parallel min/max computation
        // AsParallel() enables parallel LINQ processing
        // Reference: https://docs.microsoft.com/en-us/dotnet/standard/parallel-programming/parallel-linq-plinq
        var minX = positions.AsParallel().Min(p => p[0]);
        var maxX = positions.AsParallel().Max(p => p[0]);
        var minY = positions.AsParallel().Min(p => p[1]);
        var maxY = positions.AsParallel().Max(p => p[1]);

        var rangeX = maxX - minX;
        var rangeY = maxY - minY;

        // Handle degenerate case where all points have same coordinate
        if (rangeX < 0.0001) rangeX = 1;
        if (rangeY < 0.0001) rangeY = 1;

        // Calculate usable canvas area (5% padding on all sides)
        var padding = 0.05;
        var usableWidth = width * (1 - 2 * padding);
        var usableHeight = height * (1 - 2 * padding);

        // Pre-compute scale factors and offsets (optimization: avoid division in loop)
        // Mathematical derivation:
        // normalized = (original - min) / range * usableSize + padding
        // Rewrite to use only multiplication/addition:
        // normalized = original * (usableSize / range) + (padding - min * scale)
        double scaleX = usableWidth / rangeX;
        double scaleY = usableHeight / rangeY;
        double offsetX = width * padding - minX * scaleX;
        double offsetY = height * padding - minY * scaleY;

        var result = new double[positions.Length][];

        // Apply affine transformation in parallel
        Parallel.For(0, positions.Length, i =>
        {
            result[i] = new[]
            {
                positions[i][0] * scaleX + offsetX,  // Only multiply-add (fast)
                positions[i][1] * scaleY + offsetY
            };
        });

        return result;
    }
}