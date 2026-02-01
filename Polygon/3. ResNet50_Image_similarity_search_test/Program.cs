//SEE: 1. ResNet50_GetLogits_Test
//SEE: 2. ResNet50_GetEmbedding_Test
//BEFORE this step

namespace ResNet50_Image_similarity_search_test;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

static class Constants
{
    //"cut_to_embedded.py to cut logits layer from resnet50-v2-7.onnx
    public const string ModelPath = "resnet50-embedding-only.onnx";
    public const int ImageSize = 224;
    public const string OutputLayerName = "resnetv24_pool1_fwd";

    //public const string ModelPath = "resnet50-v2-7.onnx";  
    //public const string OutputLayerName = "resnetv24_dense0_fwd";

}

public class ImageEmbedding
{
    public string imageFile;
    public float[] ebedding;
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

        using var results = session?.Run (inputs);

        return new ImageEmbedding()
        {
            imageFile = imageFileName,
            ebedding = results?.First(r => r?.Name == Constants.OutputLayerName)?.AsEnumerable<float>()?.ToArray()
        };
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        var dotProduct = 0f;
        var magnitudeA = 0f;
        var magnitudeB = 0f;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        return dotProduct / (MathF.Sqrt(magnitudeA) * MathF.Sqrt(magnitudeB));
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
        embeddings.Add(GetEmbedding(@"Assets\3.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4001.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4002.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4003.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\cat.4004.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4081.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4082.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4083.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\dog.4084.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\1.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\2.jpg"));
        embeddings.Add(GetEmbedding(@"Assets\3 - Copy.jpg"));


        for (int i=1; i < embeddings.Count; i++)
        {
            Console.WriteLine($"Embedding {embeddings[i].imageFile} = {CosineSimilarity(embeddings[0].ebedding, embeddings[i].ebedding)}");
        }

        Console.ReadLine();

    }
}




