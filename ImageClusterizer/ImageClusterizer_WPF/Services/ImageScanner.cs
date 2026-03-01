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
    public async IAsyncEnumerable<ScanProgress> ScanFolderAsync(string folder, [EnumeratorCancellation] CancellationToken ct = default)
    {
        List<string> imageFiles = null;
        try
        {
            EnumerationOptions options = new EnumerationOptions
            {
                IgnoreInaccessible = true,
                RecurseSubdirectories = true,
                AttributesToSkip = FileAttributes.System | FileAttributes.Hidden,
                ReturnSpecialDirectories = false
            };

            imageFiles = Directory.EnumerateFiles(folder, "*.*", options)
                .Where(f => Utility.IsImageFile(f))
                .ToList();

        }
        catch (Exception e)
        {
            //user can not has right to selected folder or subfolder
            //TODO: more selective search with skip some folders
            Debug.WriteLine($"Can't open selected folder: {e.Message}");
        }

        int totalCount = imageFiles.Count;
        int processedCount = 0;

        //Producer Consumer pattern realisation for images processing with Channel using 

        //Channel for processing files
        //it is bounded channel by BoundedChannelOptions
        var fileChannel = Channel.CreateBounded<string>(
            new BoundedChannelOptions(Environment.ProcessorCount * 2)
            {
                //if more than user PC Proccesors * 2 item in channel - wait before write next item
                FullMode = BoundedChannelFullMode.Wait
            });

        // Channel for lazy return processed image via yield return
        // no bounds for this channel 
        var progressChannel = Channel.CreateUnbounded<ScanProgress>();

        //just wrote images file names to fileChannel one by one 
        var producer = Task.Run(async () =>
        {
            foreach (var imageFile in imageFiles)
            {
                if (ct.IsCancellationRequested) break;

                //Check database exists here
                if (await vectorDatabase.ExistsAsync(imageFile))
                {
                    Interlocked.Increment(ref processedCount);
                    continue;
                }

                await fileChannel.Writer.WriteAsync(imageFile);
            }
            fileChannel.Writer.Complete();
        },
        ct);

        //consumer read files names from channel and process them in 
        //user PC ProcessorCount Tasks... so if user has 2 processors only
        //2 threads process images, if 4 then 4 threads
        var consumer = Enumerable.Range(0, Environment.ProcessorCount)
            .Select(_=>Task.Run(async () =>
            {
                await foreach (var imageFile in fileChannel.Reader.ReadAllAsync(ct))
                {
                    try
                    {
                        // CNN + ResNet here
                        // Get embedding vector
                        var vector = await vectorService.GetEmbeddingAsync(imageFile);

                        var imageVector = new ImageVector
                        {
                            FilePath = imageFile,
                            Vector = vector,
                            ProcessedAt = DateTime.UtcNow,
                            FileSize = new FileInfo(imageFile).Length
                        };

                        // Save to DB
                        await vectorDatabase.SaveAsync(imageVector);

                        var count = Interlocked.Increment(ref processedCount);


                        await progressChannel.Writer.WriteAsync(new ScanProgress
                        {
                            CurrentFile = imageFile,
                            ProcessedCount = count,
                            TotalCount = totalCount,
                            NewVector = imageVector
                        }, ct);

                    }
                    catch (Exception ex)
                    {
                        //TODO: remeber select store to data base and skip the files
                        Debug.WriteLine($"Error processing {imageFile}: {ex.Message}");
                        continue;
                    }
                }
            }, ct))
            .ToArray(); //collect all tasks to array 

        var completionTask = Task.Run(async () =>
        { 
            //waiting before all consumer task is completed 
            //TODO: ASC: maybe use cancelation token or timeout here 
            await Task.WhenAll(consumer);
            progressChannel.Writer.Complete();
        });

        await foreach (var progress in progressChannel.Reader.ReadAllAsync(ct))
        {
            //Lazy return to caller all processed images 
            yield return progress;
        }

        //exit when all consumer task is done 
        await completionTask;

        //COOL!!!
    }
}

