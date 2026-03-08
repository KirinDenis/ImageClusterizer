namespace ImageClusterizer.Services;
using ImageClusterizer.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>
/// Database interface for storing and retrieving image vectors and cached PCA coordinates.
/// Implements IAsyncDisposable to allow clean shutdown (e.g. before deleting the database file).
/// </summary>
public interface IVectorDatabase : IAsyncDisposable
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

    /// <summary>
    /// Clears cached PCA coordinates (PcaX, PcaY) for all records.
    /// Forces full SVD recompute on next load.
    /// Called by RecalculatePcaCommand.
    /// </summary>
    Task ClearPcaCacheAsync();

    /// <summary>
    /// Closes and disposes the underlying database connection.
    /// Required before deleting the database file on disk.
    /// After this call the instance is unusable — call ReopenAsync to reconnect.
    /// </summary>
    Task CloseAsync();

    /// <summary>
    /// Reopens the database connection after CloseAsync.
    /// Called after ClearAllData completes to restore normal operation.
    /// </summary>
    Task ReopenAsync(string dbPath);
}
