
using ImageClusterizer.Models;
using System;
using System.Collections.Generic;

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
}