//SEE: 1. ResNet50_GetLogits_Test
//SEE: 2. ResNet50_GetEmbedding_Test
//SEE: 3. ResNet50_Image_similarity_search_test
//DON'T SEE: 4
//SEE: 5. ResNet50_Sparse_Dot_Product_test
//BEFORE this step

// Sparse dot product 

namespace ResNet50_Image_similarity_search_test;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Threading;

static class Constants
{
    //2. ResNet50_GetEmbedding_Test -> cut_to_embedded.py to cut logits layer from resnet50-v2-7.onnx
    public const string ModelPath = "resnet50-embedding-only.onnx";
    //  Use this for category (logits) results
    //      public const string ModelPath = "resnet50-v2-7.onnx";

    public const string OutputLayerName = "resnetv24_pool1_fwd";
    //  Use this for category (logits) results
    //      public const string OutputLayerName = "resnetv24_dense0_fwd";

    public const int ImageSize = 224;

    public const int sparseSize = 10;

}

public class ParallelResNetInference : IDisposable
{
    private readonly InferenceSession session;
    private readonly SemaphoreSlim _semaphore;
    private readonly int _maxParallelism;

    public ParallelResNetInference(string modelPath, int maxParallelism = 24)
    {
        _maxParallelism = maxParallelism;
        _semaphore = new SemaphoreSlim(maxParallelism, maxParallelism);

        var options = new SessionOptions
        {
            InterOpNumThreads = maxParallelism,
            IntraOpNumThreads = 8,
            ExecutionMode = ExecutionMode.ORT_PARALLEL

            //InterOpNumThreads = 1,  
            //IntraOpNumThreads = 1,  
            //ExecutionMode = ExecutionMode.ORT_SEQUENTIAL

        };

        session = new InferenceSession(modelPath, options);
    }

    /// <summary>
    ///  batch + PLinq
    /// </summary>
    public IEnumerable<ImageEmbedding> RunParallel(IEnumerable<string> imageFiles)
    {
        return imageFiles
            .AsParallel()
            .WithDegreeOfParallelism(_maxParallelism)
            .WithMergeOptions(ParallelMergeOptions.NotBuffered)
            .Select(imageFile => RunSingle(imageFile))
            .ToList();
    }

    /// <summary>
    /// Single
    /// </summary>
    private ImageEmbedding RunSingle(string imageFile)
    {

        using var image = Image.Load<Rgb24>(imageFile);

        image.Mutate(x => x.Resize(Constants.ImageSize, Constants.ImageSize));
        var input = new DenseTensor<float>(new[] { 1, 3, Constants.ImageSize, Constants.ImageSize });


        image.ProcessPixelRows(accessor =>
        {

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgb24 pixel = ref pixelRow[x];

                    input[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];
                    input[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];
                    input[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];
                }
            }
        });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("data", input)
        };

        using var results = session.Run(inputs);

        return new ImageEmbedding()
        {
            imageFile = imageFile,
            ebedding = results?.First(r => r?.Name == Constants.OutputLayerName)?.AsEnumerable<float>()?.ToArray()
        };

    }


    /*
    /// <summary>
    /// Async with SemaphoreSlim 
    /// </summary>
    public async Task<IReadOnlyList<float[]>> RunParallelAsync(
        IReadOnlyList<float[]> inputs,
        CancellationToken ct = default)
    {
        var tasks = inputs.Select(async (input, index) =>
        {
            await _semaphore.WaitAsync(ct);
            try
            {
                return await Task.Run(() => RunSingle(input), ct);
            }
            finally
            {
                _semaphore.Release();
            }
        });

        return await Task.WhenAll(tasks);
    }
    */

    public void Dispose()
    {
        session.Dispose();
        _semaphore.Dispose();
    }
}



public class ImageEmbedding
{
    public string imageFile { get; set; }
    public float[] ebedding { get; set; }
    public Dictionary<int, float> sparse { get; set; }
}

internal class Program
{
    private static InferenceSession? session;
    public static ImageEmbedding GetEmbedding(string imageFileName)
    {
        using var image = Image.Load<Rgb24>(imageFileName);

        image.Mutate(x => x.Resize(Constants.ImageSize, Constants.ImageSize));
        var input = new DenseTensor<float>(new[] { 1, 3, Constants.ImageSize, Constants.ImageSize });


        image.ProcessPixelRows(accessor =>
        {

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgb24 pixel = ref pixelRow[x];

                    input[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];
                    input[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];
                    input[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];
                }
            }
        });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("data", input)
        };

        using var results = session?.Run(inputs);

        return new ImageEmbedding()
        {
            imageFile = imageFileName,
            ebedding = results?.First(r => r?.Name == Constants.OutputLayerName)?.AsEnumerable<float>()?.ToArray()
        };
    }

    private static float Sparse_CosineSimilarity(Dictionary<int, float> a, Dictionary<int, float> b)
    {
        // Accumulator for dot product (A · B)
        var dotProduct = 0f;

        // ||A||² = sum of squared elements
        var magnitudeA = 0f;

        // ||B||² = sum of squared elements
        var magnitudeB = 0f;

        foreach (var itemA in a)
        {
            magnitudeA += itemA.Value * itemA.Value;

            //for sparse method - only if the B vector had value for the A position
            //it means:
            //if vector A had big value at position 134 (in 1000D) and B had big value to, 
            //only at this situation calculate dotProduct
            //all other position for B like 0.0f
            //lost information but works not with all vectors valueses
            if (b.TryGetValue(itemA.Key, out var itemBValue))
            {
                dotProduct += itemA.Value * itemBValue;
            }
        }

        foreach (var itemB in b)
        {
            magnitudeB += itemB.Value * itemB.Value;
        }

        if (magnitudeA == 0 || magnitudeB == 0) return 0f;

        // Final calculation: cos(θ) = dot_product / (||A|| × ||B||)
        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
    }

    public static bool IsImageFile(string filePath)
    {
        string extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension is ".jpg" or ".jpeg" or ".gif" or ".png" or ".bmp";
    }


    static void Main(string[] args)
    {
        Console.Clear();

        //--- Selected folder images test 
        var sw = Stopwatch.StartNew();
        EnumerationOptions options = new EnumerationOptions
        {
            IgnoreInaccessible = true,
            RecurseSubdirectories = true,
            AttributesToSkip = FileAttributes.System | FileAttributes.Hidden,
            ReturnSpecialDirectories = false
        };

        //Selet path to folder with images 
        //                                     CHANGE THE -> E:\CatsAndDogs\ to your local images location
        List<string> imageFiles = Directory.EnumerateFiles(@"E:\CatsAndDogs\", "*.*", options)
            .Where(f => IsImageFile(f))
            .ToList();

        Console.WriteLine($"{imageFiles.Count} files found - {sw.ElapsedMilliseconds} ms");

        //ENDOF Selected folder images test ---

        //--- Embedding 
        sw.Restart();

        // Инициализация
        using var inference = new ParallelResNetInference(Constants.ModelPath, maxParallelism: 4);

        // --- 1: PLinq (sync) ---
        var results = inference.RunParallel(imageFiles);

        // --- 2: Async + Semaphore (async) ---
        //var results = await inference.RunParallelAsync(images.ToList(), cancellationToken);

        Console.WriteLine($"vectors complete for {imageFiles.Count} image files - {sw.ElapsedMilliseconds} ms");

        //ENDOF Embedding ---

        //--- Sparse vectors 
        sw.Restart();

        foreach (var imageEmbeding in results)
        {
            //sparse dense tensor for 2048D or 1000d to 10 elements with stored positions in original vector
            imageEmbeding.sparse = imageEmbeding.ebedding
                .Select((value, index) => (index, value))
                .OrderByDescending(x => x.value)
                .Take(Constants.sparseSize)
                .ToDictionary(x => x.index, x => x.value);
        }
        Console.WriteLine($"sparse vectors calculated - {sw.ElapsedMilliseconds} ms");

        //ENDOF Sparse vectors ---

        //--- Dot Product vectors 
        sw.Restart();

        //SEE: 3. ResNet50_Image_similarity_search_test
        //SEE: SEE: 5. ResNet50_Sparse_Dot_Product_test
        //All images to dot product to first

        var resultsList = results.ToArray();

        for (int i = 1; i < resultsList.Length; i++)
        {
            Console.WriteLine($"Embedding {resultsList[i].imageFile} = {Sparse_CosineSimilarity(resultsList[0].sparse, resultsList[i].sparse)}");
        }
        Console.WriteLine($"dot product first image to all completed - {sw.ElapsedMilliseconds} ms");

        //ENDOF Dot Product vectors 

        Console.ReadLine();
    }
}
