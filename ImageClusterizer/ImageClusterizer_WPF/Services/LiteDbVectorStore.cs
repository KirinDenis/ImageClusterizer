namespace ImageClusterizer.Services;
using ImageClusterizer.Models;
using LiteDB;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class LiteDbVectorStore : IVectorDatabase
{
    private LiteDatabase? _db;
    private ILiteCollection<ImageVectorEntity>? _collection;
    private string _dbPath;

    public LiteDbVectorStore(string dbPath)
    {
        _dbPath = dbPath;
        Open();
    }

    private void Open()
    {
        _db = new LiteDatabase(_dbPath);
        _collection = _db.GetCollection<ImageVectorEntity>("vectors");
        _collection.EnsureIndex(x => x.FilePath, unique: true);
    }

    // IAsyncDisposable implementation
    public async ValueTask DisposeAsync()
    {
        await CloseAsync();
    }

    /// <summary>
    /// Closes and disposes the LiteDB connection.
    /// Must be called before attempting to delete the database file on disk.
    /// </summary>
    public Task CloseAsync()
    {
        return Task.Run(() =>
        {
            _collection = null;
            _db?.Dispose();
            _db = null;
        });
    }

    /// <summary>
    /// Reopens the database connection after CloseAsync.
    /// Called after ClearAllData completes to restore normal operation.
    /// </summary>
    public Task ReopenAsync(string dbPath)
    {
        return Task.Run(() =>
        {
            _dbPath = dbPath;
            Open();
        });
    }

    public async Task SaveAsync(ImageVector vector)
    {
        await Task.Run(() =>
        {
            var entity = new ImageVectorEntity
            {
                FilePath = vector.FilePath,
                Vector = vector.Vector,
                VectorType = vector.VectorType,
                ProcessedAt = vector.ProcessedAt,
                FileSize = vector.FileSize,
                ThumbnailPath = vector.ThumbnailPath,
                PcaX = vector.PcaX,
                PcaY = vector.PcaY
            };
            _collection!.Upsert(entity);
        });
    }

    public async Task<List<ImageVector>> GetAllAsync()
    {
        return await Task.Run(() =>
        {
            return _collection!.FindAll()
                .Select(e => new ImageVector
                {
                    FilePath = e.FilePath,
                    Vector = e.Vector,
                    VectorType = e.VectorType,
                    ProcessedAt = e.ProcessedAt,
                    FileSize = e.FileSize,
                    ThumbnailPath = e.ThumbnailPath,
                    PcaX = e.PcaX,
                    PcaY = e.PcaY
                })
                .ToList();
        });
    }

    public async Task<bool> ExistsAsync(string filePath)
    {
        return await Task.Run(() => _collection!.Exists(x => x.FilePath == filePath));
    }

    public async Task SavePcaCoordinatesAsync(string filePath, float pcaX, float pcaY)
    {
        await Task.Run(() =>
        {
            var entity = _collection!.FindOne(x => x.FilePath == filePath);
            if (entity != null)
            {
                entity.PcaX = pcaX;
                entity.PcaY = pcaY;
                _collection.Update(entity);
            }
        });
    }

    public async Task ClearPcaCacheAsync()
    {
        await Task.Run(() =>
        {
            var all = _collection!.FindAll().ToList();
            foreach (var entity in all)
            {
                entity.PcaX = null;
                entity.PcaY = null;
            }
            foreach (var entity in all)
            {
                _collection.Update(entity);
            }
        });
    }
}

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
