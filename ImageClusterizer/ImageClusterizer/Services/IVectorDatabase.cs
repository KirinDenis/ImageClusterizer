namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>
/// Database interface for storing and retrieving image vectors
/// </summary>
public interface IVectorDatabase
{
    /// <summary>
    /// Saves an image vector to the database
    /// </summary>
    Task SaveAsync(ImageVector vector);

    /// <summary>
    /// Retrieves all stored image vectors
    /// </summary>
    Task<List<ImageVector>> GetAllAsync();

    /// <summary>
    /// Checks if a vector exists for the specified file path
    /// </summary>
    Task<bool> ExistsAsync(string filePath);
}