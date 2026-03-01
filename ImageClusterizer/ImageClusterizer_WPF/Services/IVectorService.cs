namespace ImageClusterizer.Services;

using System.Threading.Tasks;

/// <summary>
/// Service for generating embedding vectors from images using ML models
/// </summary>
public interface IVectorService
{
    /// <summary>
    /// Generates an embedding vector for the specified image
    /// </summary>
    Task<float[]> GetEmbeddingAsync(string imagePath);
}