namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using ImageClusterizer.Utlility;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

public class ImageScanner
{
    private readonly IVectorDatabase vectorDatabase;
    private readonly IVectorService vectorService;
    private readonly StorageService storageService;

    private const int ThumbnailSize    = 224;
    private const int ThumbnailQuality = 85;

    public ImageScanner(
        IVectorDatabase vectorDatabase,
        IVectorService vectorService,
        StorageService storageService)
    {
        this.vectorDatabase = vectorDatabase;
        this.vectorService  = vectorService;
        this.storageService = storageService;
    }

    /// <summary>
    /// Scans a folder for images, extracts vectors, saves thumbnails and persists to database.
    /// Uses Channel-based producer/consumer pattern for parallel batch processing.
    /// Thumbnails (224x224 JPEG) are saved to the thumbnails cache folder during scan.
    ///
    /// Progress protocol:
    ///   TotalCount == -1  -> folder enumeration in progress (IsFolderScanning hint for UX)
    ///   TotalCount ==  0  -> enumeration done, no new images found
    ///   TotalCount  >  0  -> normal scan progress
    /// </summary>
    public async IAsyncEnumerable<ScanProgress> ScanFolderAsync(
        string folder,
        VectorType vectorType = VectorType.Embedding,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        // ---- Phase 0: signal UX that we are enumerating the folder (may be slow on large drives) ----
        yield return new ScanProgress
        {
            CurrentFile    = folder,
            ProcessedCount = 0,
            TotalCount     = -1  // sentinel: folder enumeration in progress
        };

        // ---- Phase 1: enumerate files on a background thread (avoids UI freeze) ----
        List<string> imageFiles = new();
        try
        {
            imageFiles = await Task.Run(() =>
            {
                var options = new EnumerationOptions
                {
                    IgnoreInaccessible       = true,
                    RecurseSubdirectories    = true,
                    AttributesToSkip         = FileAttributes.System | FileAttributes.Hidden,
                    ReturnSpecialDirectories = false
                };
                return Directory.EnumerateFiles(folder, "*.*", options)
                    .Where(f => Utility.IsImageFile(f))
                    .ToList();
            }, ct);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"Can't open selected folder: {e.Message}");
        }

        int totalCount     = imageFiles.Count;
        int processedCount = 0;

        if (totalCount == 0)
        {
            yield return new ScanProgress
            {
                CurrentFile    = folder,
                ProcessedCount = 0,
                TotalCount     = 0
            };
            yield break;
        }

        // ---- Phase 2: process images via bounded channel ----
        var fileChannel = Channel.CreateBounded<string>(
            new BoundedChannelOptions(Environment.ProcessorCount * 2)
            {
                FullMode = BoundedChannelFullMode.Wait
            });

        var progressChannel = Channel.CreateUnbounded<ScanProgress>();

        var producer = Task.Run(async () =>
        {
            foreach (var imageFile in imageFiles)
            {
                if (ct.IsCancellationRequested) break;
                if (await vectorDatabase.ExistsAsync(imageFile))
                {
                    Interlocked.Increment(ref processedCount);
                    continue;
                }
                await fileChannel.Writer.WriteAsync(imageFile);
            }
            fileChannel.Writer.Complete();
        }, ct);

        var consumers = Enumerable.Range(0, Environment.ProcessorCount)
            .Select(_ => Task.Run(async () =>
            {
                await foreach (var imageFile in fileChannel.Reader.ReadAllAsync(ct))
                {
                    try
                    {
                        string? thumbnailPath = null;
                        using (var image = Image.Load<Rgb24>(imageFile))
                        {
                            image.Mutate(x => x.Resize(new ResizeOptions
                            {
                                Size = new Size(ThumbnailSize, ThumbnailSize),
                                Mode = ResizeMode.Crop
                            }));
                            var thumbPath = storageService.GetThumbnailPath(imageFile);
                            if (!File.Exists(thumbPath))
                            {
                                await image.SaveAsJpegAsync(thumbPath,
                                    new JpegEncoder { Quality = ThumbnailQuality });
                            }
                            thumbnailPath = thumbPath;
                        }

                        var vector = await vectorService.GetEmbeddingAsync(imageFile, vectorType);
                        var imageVector = new ImageVector
                        {
                            FilePath      = imageFile,
                            Vector        = vector,
                            VectorType    = vectorType,
                            ProcessedAt   = DateTime.UtcNow,
                            FileSize      = new FileInfo(imageFile).Length,
                            ThumbnailPath = thumbnailPath
                        };
                        await vectorDatabase.SaveAsync(imageVector);

                        var count = Interlocked.Increment(ref processedCount);
                        await progressChannel.Writer.WriteAsync(new ScanProgress
                        {
                            CurrentFile    = imageFile,
                            ProcessedCount = count,
                            TotalCount     = totalCount,
                            NewVector      = imageVector
                        }, ct);
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error processing {imageFile}: {ex.Message}");
                    }
                }
            }, ct))
            .ToArray();

        var completionTask = Task.Run(async () =>
        {
            await Task.WhenAll(consumers);
            progressChannel.Writer.Complete();
        });

        await foreach (var progress in progressChannel.Reader.ReadAllAsync(ct))
        {
            yield return progress;
        }

        await completionTask;
    }
}
