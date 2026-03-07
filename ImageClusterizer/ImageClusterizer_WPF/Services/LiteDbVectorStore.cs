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
                FilePath    = vector.FilePath,
                Vector      = vector.Vector,
                VectorType  = vector.VectorType,
                ProcessedAt = vector.ProcessedAt,
                FileSize    = vector.FileSize
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
                    FilePath    = e.FilePath,
                    Vector      = e.Vector,
                    VectorType  = e.VectorType,
                    ProcessedAt = e.ProcessedAt,
                    FileSize    = e.FileSize
                })
                .ToList();
        });
    }

    public async Task<bool> ExistsAsync(string filePath)
    {
        return await Task.Run(() => _collection.Exists(x => x.FilePath == filePath));
    }
}

/// <summary>
/// LiteDB entity for persisting image vectors.
/// Mirrors ImageVector model with all fields including VectorType and FileSize.
/// </summary>
public class ImageVectorEntity
{
    public ObjectId Id { get; set; }
    public string FilePath { get; set; }
    public float[] Vector { get; set; }
    public VectorType VectorType { get; set; }
    public DateTime ProcessedAt { get; set; }
    public long FileSize { get; set; }
}
