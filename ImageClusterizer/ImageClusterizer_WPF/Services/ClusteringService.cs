using ImageClusterizer.Models;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

public class ClusteringService
{
    public List<ImageCluster> ClusterBySimilarity(
        List<ImageVector> vectors,
        float similarityThreshold = 0.85f)
    {
        var clusters = new List<ImageCluster>();
        var assigned = new HashSet<string>();

        foreach (var vector in vectors)
        {
            if (assigned.Contains(vector.FilePath)) continue;

            var cluster = new ImageCluster
            {
                ClusterId = clusters.Count,
                Images = new List<ImageVector> { vector }
            };
            assigned.Add(vector.FilePath);

            foreach (var candidate in vectors)
            {
                if (assigned.Contains(candidate.FilePath)) continue;
                var similarity = CosineSimilarity(vector.Vector, candidate.Vector);
                if (similarity >= similarityThreshold)
                {
                    cluster.Images.Add(candidate);
                    assigned.Add(candidate.FilePath);
                }
            }

            cluster.Centroid = CalculateCentroid(cluster.Images);
            clusters.Add(cluster);
        }

        return clusters;
    }

    private float CosineSimilarity(float[] a, float[] b)
    {
        var dotProduct = 0f;
        var magnitudeA = 0f;
        var magnitudeB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }
        var denom = MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB);
        if (denom < 1e-10f) return 0f;
        return dotProduct / denom;
    }

    private float[] CalculateCentroid(List<ImageVector> vectors)
    {
        var dimension = vectors[0].Vector.Length;
        var centroid = new float[dimension];
        foreach (var vector in vectors)
            for (int i = 0; i < dimension; i++)
                centroid[i] += vector.Vector[i];
        for (int i = 0; i < dimension; i++)
            centroid[i] /= vectors.Count;
        return centroid;
    }

    public List<ClusterPosition> CalculatePositions(
        List<ImageCluster> clusters,
        int canvasWidth = 10000,
        int canvasHeight = 10000)
    {
        var allVectors = new List<float[]>();
        var vectorInfo = new List<VectorInfo>();

        foreach (var cluster in clusters)
        {
            if (cluster.Centroid != null)
            {
                allVectors.Add(cluster.Centroid);
                vectorInfo.Add(new VectorInfo { ClusterId = cluster.ClusterId, IsCentroid = true, ImageVector = null });
            }
            foreach (var image in cluster.Images)
            {
                allVectors.Add(image.Vector);
                vectorInfo.Add(new VectorInfo { ClusterId = cluster.ClusterId, IsCentroid = false, ImageVector = image });
            }
        }

        if (allVectors.Count == 0) return new List<ClusterPosition>();

        var positions2D = ReduceTo2D_PCA(allVectors);
        var normalized = NormalizePositions(positions2D, canvasWidth, canvasHeight);

        var result = new List<ClusterPosition>();
        for (int i = 0; i < normalized.Length; i++)
        {
            result.Add(new ClusterPosition
            {
                ClusterId = vectorInfo[i].ClusterId,
                IsCentroid = vectorInfo[i].IsCentroid,
                ImageVector = vectorInfo[i].ImageVector,
                X = normalized[i][0],
                Y = normalized[i][1]
            });
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Sparse compression - based on Polygon/5 ResNet50_Sparse_Dot_Product_test
    // -------------------------------------------------------------------------

    /// <summary>
    /// Converts a dense embedding to sparse representation by keeping only top-N values.
    /// sparseTopN = 2048 means no compression (full vector returned as-is).
    /// sparseTopN = 10 means extreme compression (10 out of 2048 values kept, rest zeroed).
    /// </summary>
    public static float[] ToSparse(float[] vector, int sparseTopN)
    {
        if (sparseTopN <= 0 || sparseTopN >= vector.Length) return vector;
        var result = new float[vector.Length];
        var topIndices = vector
            .Select((v, i) => (index: i, absValue: MathF.Abs(v)))
            .OrderByDescending(x => x.absValue)
            .Take(sparseTopN)
            .Select(x => x.index);
        foreach (var idx in topIndices)
            result[idx] = vector[idx];
        return result;
    }

    /// <summary>
    /// CalculatePositions variant that applies sparse compression before PCA.
    /// Uses Randomized PCA (RSVD) instead of full SVD to handle very large datasets
    /// (100k+ vectors) without arithmetic overflow or excessive memory use.
    /// Full SVD on 200k x 2048 would require ~3.2 GB RAM and causes overflow.
    /// RSVD uses O(n*k) memory where k=12 (2 target + 10 oversampling).
    /// </summary>
    public List<ClusterPosition> CalculatePositionsSparse(
        List<ImageCluster> clusters,
        int canvasWidth,
        int canvasHeight,
        int sparseTopN,
        IProgress<(int current, int total, string message)>? progress = null)
    {
        var allVectors = new List<float[]>();
        var vectorInfo = new List<VectorInfo>();

        foreach (var cluster in clusters)
        {
            if (cluster.Centroid != null)
            {
                allVectors.Add(cluster.Centroid);
                vectorInfo.Add(new VectorInfo { ClusterId = cluster.ClusterId, IsCentroid = true, ImageVector = null });
            }
            foreach (var image in cluster.Images)
            {
                allVectors.Add(image.Vector);
                vectorInfo.Add(new VectorInfo { ClusterId = cluster.ClusterId, IsCentroid = false, ImageVector = image });
            }
        }

        if (allVectors.Count == 0) return new List<ClusterPosition>();

        int total = allVectors.Count;

        // Apply sparse compression to each vector
        var compressedVectors = new List<float[]>(total);
        for (int i = 0; i < total; i++)
        {
            compressedVectors.Add(ToSparse(allVectors[i], sparseTopN));
            if (i % 500 == 0)
                progress?.Report((i, total, $"Compressing vectors: {i}/{total} (Top-{sparseTopN})"));
        }
        progress?.Report((total, total, $"Compression done - computing PCA on {total} vectors..."));

        // Randomized PCA handles 100k-500k vectors without overflow
        var positions2D = ReduceTo2D_RandomizedPCA(compressedVectors, progress, total);
        progress?.Report((total, total, "PCA complete - normalizing positions..."));

        var normalized = NormalizePositions(positions2D, canvasWidth, canvasHeight);

        var result = new List<ClusterPosition>();
        for (int i = 0; i < normalized.Length; i++)
        {
            result.Add(new ClusterPosition
            {
                ClusterId = vectorInfo[i].ClusterId,
                IsCentroid = vectorInfo[i].IsCentroid,
                ImageVector = vectorInfo[i].ImageVector,
                X = normalized[i][0],
                Y = normalized[i][1]
            });
        }
        progress?.Report((total, total, $"Positions ready - {result.Count} points"));
        return result;
    }

    // -------------------------------------------------------------------------
    // Randomized PCA (RSVD) - handles 100k-500k vectors without overflow
    // -------------------------------------------------------------------------

    /// <summary>
    /// Randomized SVD via power iteration (Halko et al. 2011).
    /// Memory footprint: O(n*l + d*l) where l=12 (k=2 + oversampling=10).
    /// Compared to full SVD: O(n*d) = O(200000*2048) = ~3.2 GB for double.
    /// Provides good approximation for top-2 principal components.
    /// powerIterations=3 gives excellent accuracy at moderate extra cost.
    /// </summary>
    private double[][] ReduceTo2D_RandomizedPCA(
        List<float[]> vectors,
        IProgress<(int current, int total, string message)>? progress = null,
        int total = 0)
    {
        int n = vectors.Count;
        int d = vectors[0].Length;

        if (n < 2 || d < 2)
            return Enumerable.Range(0, n).Select(_ => new double[] { 0.0, 0.0 }).ToArray();

        // Step 1: Compute column means for centering
        progress?.Report((0, total, "RSVD: Computing column means..."));
        var means = new double[d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                means[j] += vectors[i][j];
        for (int j = 0; j < d; j++)
            means[j] /= n;

        // k = target rank, p = oversampling, l = sketch size
        int k = 2;
        int p = 10;
        int l = k + p;

        // Step 2: Random Gaussian matrix Omega (d x l)
        progress?.Report((0, total, $"RSVD: Building sketch ({n} x {d} -> {n} x {l})..."));
        var rng = new Random(42);
        var omega = new double[d, l];
        for (int j = 0; j < d; j++)
            for (int s = 0; s < l; s++)
                omega[j, s] = SampleGaussian(rng);

        // Step 3: Y = A_centered * Omega  (n x l)
        progress?.Report((0, total, "RSVD: Random projection..."));
        var Y = new double[n, l];
        for (int i = 0; i < n; i++)
        {
            for (int s = 0; s < l; s++)
            {
                double dot = 0.0;
                for (int j = 0; j < d; j++)
                    dot += (vectors[i][j] - means[j]) * omega[j, s];
                Y[i, s] = dot;
            }
            if (i % 10000 == 0 && total > 0)
                progress?.Report((i, total, $"RSVD: Projecting row {i}/{n}..."));
        }

        // Step 4: Power iterations for accuracy improvement
        int powerIterations = 3;
        for (int iter = 0; iter < powerIterations; iter++)
        {
            progress?.Report((0, total, $"RSVD: Power iteration {iter + 1}/{powerIterations}..."));

            // At = A_centered^T * Y  (d x l)
            var At = new double[d, l];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                {
                    double val = vectors[i][j] - means[j];
                    for (int s = 0; s < l; s++)
                        At[j, s] += val * Y[i, s];
                }

            // Y = A_centered * At  (n x l)
            var Y2 = new double[n, l];
            for (int i = 0; i < n; i++)
                for (int s = 0; s < l; s++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < d; j++)
                        dot += (vectors[i][j] - means[j]) * At[j, s];
                    Y2[i, s] = dot;
                }
            Y = Y2;
        }

        // Step 5: QR decomposition of Y  -> Q (n x l)
        progress?.Report((0, total, "RSVD: QR decomposition..."));
        var Ymat = Matrix<double>.Build.DenseOfArray(Y);
        var qr = Ymat.QR();
        var Q = qr.Q;

        // Step 6: B = Q^T * A_centered  (l x d) - small matrix, safe SVD
        progress?.Report((0, total, "RSVD: Small matrix projection..."));
        var B = new double[l, d];
        for (int s = 0; s < l; s++)
            for (int i = 0; i < n; i++)
            {
                double q = Q[i, s];
                if (q == 0.0) continue;
                for (int j = 0; j < d; j++)
                    B[s, j] += q * (vectors[i][j] - means[j]);
            }

        // Step 7: SVD of small B  (l=12 x d, negligible)
        progress?.Report((0, total, "RSVD: Small SVD (l x d sketch)..."));
        var Bmat = Matrix<double>.Build.DenseOfArray(B);
        var svd = Bmat.Svd(computeVectors: true);
        var Vt = svd.VT;

        // Step 8: Project to 2D using top-2 right singular vectors
        progress?.Report((0, total, "RSVD: Final 2D projection..."));
        var result = new double[n][];
        for (int i = 0; i < n; i++)
            result[i] = new double[2];

        for (int pc = 0; pc < 2 && pc < Vt.RowCount; pc++)
            for (int i = 0; i < n; i++)
            {
                double proj = 0.0;
                for (int j = 0; j < d; j++)
                    proj += (vectors[i][j] - means[j]) * Vt[pc, j];
                result[i][pc] = proj;
            }

        return result;
    }

    private static double SampleGaussian(Random rng)
    {
        // Box-Muller transform for standard normal distribution
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    // -------------------------------------------------------------------------
    // Legacy full PCA - used only when n <= 5000 (safe range)
    // -------------------------------------------------------------------------

    private double[][] ReduceTo2D_PCA(List<float[]> vectors)
    {
        int n = vectors.Count;
        int d = vectors[0].Length;

        if (n < 2 || d < 2)
            return Enumerable.Range(0, n).Select(_ => new double[] { 0.0, 0.0 }).ToArray();

        // Safety gate: large datasets fall through to Randomized PCA
        if (n > 5000)
            return ReduceTo2D_RandomizedPCA(vectors);

        var matrixData = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                matrixData[i, j] = vectors[i][j];

        var matrix = Matrix<double>.Build.DenseOfArray(matrixData);
        var columnMeans = matrix.ColumnSums() / n;
        var centered = matrix.Clone();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                centered[i, j] -= columnMeans[j];

        var svd = centered.Svd(computeVectors: true);
        var u = svd.U;
        var s = svd.S;

        double s0 = s.Count > 0 ? s[0] : 0.0;
        double s1 = s.Count > 1 ? s[1] : 0.0;

        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            double x = u.RowCount > i && u.ColumnCount > 0 ? u[i, 0] * s0 : 0.0;
            double y = u.RowCount > i && u.ColumnCount > 1 ? u[i, 1] * s1 : 0.0;
            result[i] = new double[] { x, y };
        }
        return result;
    }

    private double[][] NormalizePositions(double[][] positions, int width, int height)
    {
        if (positions.Length == 0) return positions;

        var minX = positions.Min(p => p[0]);
        var maxX = positions.Max(p => p[0]);
        var minY = positions.Min(p => p[1]);
        var maxY = positions.Max(p => p[1]);

        var rangeX = maxX - minX;
        var rangeY = maxY - minY;
        if (rangeX < 0.0001) rangeX = 1;
        if (rangeY < 0.0001) rangeY = 1;

        var padding = 0.05;
        var usableWidth = width * (1 - 2 * padding);
        var usableHeight = height * (1 - 2 * padding);

        return positions.Select(p => new[]
        {
            (p[0] - minX) / rangeX * usableWidth + width * padding,
            (p[1] - minY) / rangeY * usableHeight + height * padding
        }).ToArray();
    }
}
