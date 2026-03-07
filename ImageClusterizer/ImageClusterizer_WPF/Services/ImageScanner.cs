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

    private const int ThumbnailSize   = 224;
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
    /// </summary>
    public async IAsyncEnumerable<ScanProgress> ScanFolderAsync(
        string folder,
        VectorType vectorType = VectorType.Embedding,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        List<string> imageFiles = null;

        try
        {
            EnumerationOptions options = new EnumerationOptions
            {
                IgnoreInaccessible       = true,
                RecurseSubdirectories    = true,
                AttributesToSkip         = FileAttributes.System | FileAttributes.Hidden,
                ReturnSpecialDirectories = false
            };

            imageFiles = Directory.EnumerateFiles(folder, "*.*", options)
                .Where(f => Utility.IsImageFile(f))
                .ToList();
        }
        catch (Exception e)
        {
            // User may not have access rights to selected folder or subfolder
            Debug.WriteLine($"Can't open selected folder: {e.Message}");
        }

        int totalCount     = imageFiles?.Count ?? 0;
        int processedCount = 0;

        // Bounded channel: prevents unbounded memory growth when producer is faster than consumers
        var fileChannel = Channel.CreateBounded<string>(
            new BoundedChannelOptions(Environment.ProcessorCount * 2)
            {
                FullMode = BoundedChannelFullMode.Wait
            });

        // Unbounded channel for lazy return of progress updates via yield return
        var progressChannel = Channel.CreateUnbounded<ScanProgress>();

        // Producer: writes image file paths into fileChannel one by one
        var producer = Task.Run(async () =>
        {
            foreach (var imageFile in imageFiles ?? new List<string>())
            {
                if (ct.IsCancellationRequested) break;

                // Skip files already in the database
                if (await vectorDatabase.ExistsAsync(imageFile))
                {
                    Interlocked.Increment(ref processedCount);
                    continue;
                }

                await fileChannel.Writer.WriteAsync(imageFile);
            }
            fileChannel.Writer.Complete();
        }, ct);

        // Consumers: process images in parallel using ProcessorCount tasks
        var consumers = Enumerable.Range(0, Environment.ProcessorCount)
            .Select(_ => Task.Run(async () =>
            {
                await foreach (var imageFile in fileChannel.Reader.ReadAllAsync(ct))
                {
                    try
                    {
                        string? thumbnailPath = null;

                        // Load image once, preprocess to 224x224 for both ONNX and thumbnail
                        using (var image = Image.Load<Rgb24>(imageFile))
                        {
                            // Resize to 224x224 (same as ONNX preprocessing)
                            image.Mutate(x => x.Resize(new ResizeOptions
                            {
                                Size = new Size(ThumbnailSize, ThumbnailSize),
                                Mode = ResizeMode.Crop
                            }));

                            // Save thumbnail if it does not already exist
                            var thumbPath = storageService.GetThumbnailPath(imageFile);
                            if (!File.Exists(thumbPath))
                            {
                                await image.SaveAsJpegAsync(thumbPath,
                                    new JpegEncoder { Quality = ThumbnailQuality });
                            }
                            thumbnailPath = thumbPath;
                        }

                        // Extract feature vector using CNN (ResNet50)
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

                        // Persist to database
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
                        // TODO: store failed files and allow retry
                        Debug.WriteLine($"Error processing {imageFile}: {ex.Message}");
                    }
                }
            }, ct))
            .ToArray();

        // Wait for all consumers, then close progress channel
        var completionTask = Task.Run(async () =>
        {
            await Task.WhenAll(consumers);
            progressChannel.Writer.Complete();
        });

        // Lazily yield progress updates back to the caller
        await foreach (var progress in progressChannel.Reader.ReadAllAsync(ct))
        {
            yield return progress;
        }

        await completionTask;
    }
}
