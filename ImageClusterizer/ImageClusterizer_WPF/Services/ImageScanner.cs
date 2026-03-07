namespace ImageClusterizer.Services;

using ImageClusterizer.Models;
using ImageClusterizer.Utlility;
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

    public ImageScanner(IVectorDatabase vectorDatabase, IVectorService vectorService)
    {
        this.vectorDatabase = vectorDatabase;
        this.vectorService = vectorService;
    }

    /// <summary>
    /// Scans a folder for images, extracts vectors and saves them to the database.
    /// Uses Channel-based producer/consumer pattern for parallel batch processing.
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
                IgnoreInaccessible    = true,
                RecurseSubdirectories = true,
                AttributesToSkip      = FileAttributes.System | FileAttributes.Hidden,
                ReturnSpecialDirectories = false
            };

            imageFiles = Directory.EnumerateFiles(folder, "*.*", options)
                .Where(f => Utility.IsImageFile(f))
                .ToList();
        }
        catch (Exception e)
        {
            // User may not have rights to selected folder or subfolder
            // TODO: more selective search with per-folder error handling
            Debug.WriteLine($"Can't open selected folder: {e.Message}");
        }

        int totalCount = imageFiles?.Count ?? 0;
        int processedCount = 0;

        // Producer/Consumer pattern with Channel for parallel image processing
        // Bounded channel: prevents unbounded memory growth when producer is faster than consumers
        var fileChannel = Channel.CreateBounded<string>(
            new BoundedChannelOptions(Environment.ProcessorCount * 2)
            {
                // If more items than (ProcessorCount * 2) are queued - block the producer
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
        // Each task reads from fileChannel, extracts vector, saves to DB
        var consumers = Enumerable.Range(0, Environment.ProcessorCount)
            .Select(_ => Task.Run(async () =>
            {
                await foreach (var imageFile in fileChannel.Reader.ReadAllAsync(ct))
                {
                    try
                    {
                        // Extract feature vector using CNN (ResNet50)
                        var vector = await vectorService.GetEmbeddingAsync(imageFile, vectorType);

                        var imageVector = new ImageVector
                        {
                            FilePath    = imageFile,
                            Vector      = vector,
                            VectorType  = vectorType,
                            ProcessedAt = DateTime.UtcNow,
                            FileSize    = new FileInfo(imageFile).Length
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
                        // TODO: store failed files in database and allow retry
                        Debug.WriteLine($"Error processing {imageFile}: {ex.Message}");
                        continue;
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
