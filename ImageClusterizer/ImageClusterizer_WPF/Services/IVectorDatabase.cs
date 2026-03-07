namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>
/// Database interface for storing and retrieving image vectors and cached PCA coordinates
/// </summary>
public interface IVectorDatabase
{
    /// <summary>Saves a new image vector (or updates existing) in the database</summary>
    Task SaveAsync(ImageVector vector);

    /// <summary>Retrieves all stored image vectors including cached PCA coordinates</summary>
    Task<List<ImageVector>> GetAllAsync();

    /// <summary>Checks if a vector already exists for the specified file path</summary>
    Task<bool> ExistsAsync(string filePath);

    /// <summary>
    /// Persists the computed 2D PCA coordinates for a single image.
    /// Called after PCA computation to cache positions for fast startup.
    /// </summary>
    Task SavePcaCoordinatesAsync(string filePath, float pcaX, float pcaY);
}
