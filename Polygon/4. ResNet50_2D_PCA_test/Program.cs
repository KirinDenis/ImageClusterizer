//SEE: 1. ResNet50_GetLogits_Test
//SEE: 2. ResNet50_GetEmbedding_Test
//BEFORE this step

// Calculate CosineSimilarity for images using ResNet50 embeddings

namespace ResNet50_Image_similarity_search_test;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using MathNet.Numerics.LinearAlgebra;

static class Constants
{
    //2. ResNet50_GetEmbedding_Test -> cut_to_embedded.py to cut logits layer from resnet50-v2-7.onnx
    public const string ModelPath = "resnet50-embedding-only.onnx";
    //Use this for category results
    //public const string ModelPath = "resnet50-v2-7.onnx";
    
    public const string OutputLayerName = "resnetv24_pool1_fwd";
    //Use this for category results
    //public const string OutputLayerName = "resnetv24_dense0_fwd";

    public const int ImageSize = 224;

    public const int areaWidth = 80;

    public const int areaHeight = 20;

}

public class ImageEmbedding
{
    public string imageFile { get; set; }
    public float[] ebedding { get; set; }

    public int x { get; set; }

    public int y { get; set; }
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


    private static double[][] NormalizePositions(double[][] positions, int width, int height)
    {
        if (positions.Length == 0) return positions;

        var minX = positions.Min(p => p[0]);
        var maxX = positions.Max(p => p[0]);
        var minY = positions.Min(p => p[1]);
        var maxY = positions.Max(p => p[1]);

        var rangeX = maxX - minX;
        var rangeY = maxY - minY;

        if (rangeX < 0.0001) rangeX = 1;
        if (rangeY < 0.0001) rangeY = 1;

        var padding = 0.05;
        var usableWidth = width * (1 - 2 * padding);
        var usableHeight = height * (1 - 2 * padding);

        return positions.Select(p => new[]
        {
                (p[0] - minX) / rangeX * usableWidth + width * padding,
                (p[1] - minY) / rangeY * usableHeight + height * padding
            }).ToArray();
    }

    private static double[][] ReduceTo2D_PCA(List<ImageEmbedding> embeddings)
    {
        int vectors_count = embeddings.Count;
        int one_vector_dimension = embeddings[0].ebedding.Length;


        var matrixData = new double[vectors_count, one_vector_dimension];
        for (int i = 0; i < vectors_count; i++)
        {
            for (int j = 0; j < one_vector_dimension; j++)
            {
                matrixData[i, j] = embeddings[i].ebedding[j];
            }
        }

        var matrix = Matrix<double>.Build.DenseOfArray(matrixData);


        var columnMeans = matrix.ColumnSums() / vectors_count;
        var centered = matrix.Clone();

        for (int i = 0; i < vectors_count; i++)
        {
            for (int j = 0; j < one_vector_dimension; j++)
            {
                centered[i, j] -= columnMeans[j];
            }
        }

        // SVD
        //Singulat value decomposition 
        var svd = centered.Svd(computeVectors: true);
        var u = svd.U; //left singular vectors
        var s = svd.S; //singular values 


        var result = new double[vectors_count][];
        for (int i = 0; i < vectors_count; i++)
        {
            result[i] = new double[]
            {
                    u[i, 0] * s[0], //vector proection to first component
                    u[i, 1] * s[1]  //vector proection to second component
            };
        }

        return result;
    }


    static void Main(string[] args)
    {
        if (!File.Exists(Constants.ModelPath))
        {
            Console.WriteLine("Download ONNX Model Zoo from  https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx");
            Console.WriteLine("after use cut_to_embedded.py from 2. ResNet50_GetEmbedding_Test to cut logits layer");
            return;
        }

        session = new InferenceSession(Constants.ModelPath);

        List<ImageEmbedding> embeddings = new List<ImageEmbedding>();
        embeddings.Add(GetEmbedding(@"Assets\cat.4001.copy.jpg"));        
        embeddings.Add(GetEmbedding(@"Assets\cat.4001.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4002.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4003.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4004.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4081.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4082.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4083.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4084.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\noise1.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\noise2.jpg"));

        // PCA reduction
        var positions2D = ReduceTo2D_PCA(embeddings);

        var normalized = NormalizePositions(positions2D, Constants.areaWidth, Constants.areaHeight);

        Console.Clear();

        for (int i = 0; i < normalized.Length; i++)
        {
            Console.CursorLeft = (int)normalized[i][0];
            Console.CursorTop = (int) normalized[i][1];
            Console.WriteLine(Path.GetFileName(embeddings[i].imageFile));
        }


        Console.ReadLine();

    }
}




