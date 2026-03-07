namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using System.Threading.Tasks;

/// <summary>
/// Service for generating feature vectors from images using ML models
/// </summary>
public interface IVectorService
{
    /// <summary>
    /// Generates a feature vector for the specified image.
    /// Returns either an embedding (2048D) or logit vector (1000D)
    /// depending on the specified VectorType.
    /// </summary>
    Task<float[]> GetEmbeddingAsync(string imagePath, VectorType vectorType = VectorType.Embedding);
}
