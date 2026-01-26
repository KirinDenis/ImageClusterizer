
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
            if (assigned.Contains(vector.FilePath))
                continue;

            var cluster = new ImageCluster
            {
                ClusterId = clusters.Count,
                Images = new List<ImageVector> { vector }
            };

            assigned.Add(vector.FilePath);

            // Find similar
            foreach (var candidate in vectors)
            {
                if (assigned.Contains(candidate.FilePath))
                    continue;

                var similarity = CosineSimilarity(vector.Vector, candidate.Vector);

                if (similarity >= similarityThreshold)
                {
                    cluster.Images.Add(candidate);
                    assigned.Add(candidate.FilePath);
                }
            }

            // Calulate center of cluster
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

        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
    }

    private float[] CalculateCentroid(List<ImageVector> vectors)
    {
        var dimension = vectors[0].Vector.Length;
        var centroid = new float[dimension];

        foreach (var vector in vectors)
        {
            for (int i = 0; i < dimension; i++)
            {
                centroid[i] += vector.Vector[i];
            }
        }

        for (int i = 0; i < dimension; i++)
        {
            centroid[i] /= vectors.Count;
        }

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
                vectorInfo.Add(new VectorInfo
                {
                    ClusterId = cluster.ClusterId,
                    IsCentroid = true,
                    ImageVector = null
                });
            }

            foreach (var image in cluster.Images)
            {
                allVectors.Add(image.Vector);
                vectorInfo.Add(new VectorInfo
                {
                    ClusterId = cluster.ClusterId,
                    IsCentroid = false,
                    ImageVector = image
                });
            }
        }

        if (allVectors.Count == 0)
            return new List<ClusterPosition>();

        // PCA reduction
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

    private double[][] ReduceTo2D_PCA(List<float[]> vectors)
    {
        int n = vectors.Count;
        int d = vectors[0].Length;


        var matrixData = new double[n, d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                matrixData[i, j] = vectors[i][j];
            }
        }

        var matrix = Matrix<double>.Build.DenseOfArray(matrixData);


        var columnMeans = matrix.ColumnSums() / n;
        var centered = matrix.Clone();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                centered[i, j] -= columnMeans[j];
            }
        }

        // SVD
        var svd = centered.Svd(computeVectors: true);
        var u = svd.U;
        var s = svd.S;


        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[]
            {
                    u[i, 0] * s[0],
                    u[i, 1] * s[1]
            };
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
