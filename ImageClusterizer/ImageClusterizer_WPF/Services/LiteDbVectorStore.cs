namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using LiteDB;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class LiteDbVectorStore : IVectorDatabase
{
    private readonly LiteDatabase _db;
    private readonly ILiteCollection<ImageVectorEntity> _collection;

    public LiteDbVectorStore(string dbPath)
    {
        _db = new LiteDatabase(dbPath);
        _collection = _db.GetCollection<ImageVectorEntity>("vectors");
        _collection.EnsureIndex(x => x.FilePath, unique: true);
    }

    public async Task SaveAsync(ImageVector vector)
    {
        await Task.Run(() =>
        {
            var entity = new ImageVectorEntity
            {
                FilePath      = vector.FilePath,
                Vector        = vector.Vector,
                VectorType    = vector.VectorType,
                ProcessedAt   = vector.ProcessedAt,
                FileSize      = vector.FileSize,
                ThumbnailPath = vector.ThumbnailPath,
                PcaX          = vector.PcaX,
                PcaY          = vector.PcaY
            };
            _collection.Upsert(entity);
        });
    }

    public async Task<List<ImageVector>> GetAllAsync()
    {
        return await Task.Run(() =>
        {
            return _collection.FindAll()
                .Select(e => new ImageVector
                {
                    FilePath      = e.FilePath,
                    Vector        = e.Vector,
                    VectorType    = e.VectorType,
                    ProcessedAt   = e.ProcessedAt,
                    FileSize      = e.FileSize,
                    ThumbnailPath = e.ThumbnailPath,
                    PcaX          = e.PcaX,
                    PcaY          = e.PcaY
                })
                .ToList();
        });
    }

    public async Task<bool> ExistsAsync(string filePath)
    {
        return await Task.Run(() => _collection.Exists(x => x.FilePath == filePath));
    }

    /// <summary>
    /// Updates only PCA coordinates for the given file path.
    /// Does not overwrite vector data — uses a targeted update for performance.
    /// </summary>
    public async Task SavePcaCoordinatesAsync(string filePath, float pcaX, float pcaY)
    {
        await Task.Run(() =>
        {
            var entity = _collection.FindOne(x => x.FilePath == filePath);
            if (entity != null)
            {
                entity.PcaX = pcaX;
                entity.PcaY = pcaY;
                _collection.Update(entity);
            }
        });
    }
}

/// <summary>
/// LiteDB persistence entity for image vectors.
/// Mirrors ImageVector model including thumbnail path and cached PCA coordinates.
/// </summary>
public class ImageVectorEntity
{
    public ObjectId Id { get; set; }
    public string FilePath { get; set; }
    public float[] Vector { get; set; }
    public VectorType VectorType { get; set; }
    public DateTime ProcessedAt { get; set; }
    public long FileSize { get; set; }
    public string? ThumbnailPath { get; set; }
    public float? PcaX { get; set; }
    public float? PcaY { get; set; }
}
