namespace ImageClusterizer.Services;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;



public class ResNetVectorizer : IVectorService, IDisposable
{
    private readonly InferenceSession _session;
    private const int ImageSize = 224;

    // ImageNet normalization constants
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    public ResNetVectorizer(string onnxModelPath)
    {
        _session = new InferenceSession(onnxModelPath);
    }

    public async Task<float[]> GetEmbeddingAsync(string imagePath)
    {
        return await Task.Run(() =>
        {
            
            using var image = Image.Load<Rgb24>(imagePath);

            
            var inputTensor = PreprocessImage(image);

            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", inputTensor)
            };

            
            using var results = _session.Run(inputs);

            // embeddings
            // => var embedding = results.First(r => r.Name == "resnetv24_dense0_fwd").AsEnumerable<float>().ToArray();
            //                                                 ^^^ hardcoded at model  
            var output = results.First().AsEnumerable<float>().ToArray();

            return output;
        });
    }

    private DenseTensor<float> PreprocessImage(Image<Rgb24> image)
    {
        // Resize and crop to 224x224
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(ImageSize, ImageSize),
            Mode = ResizeMode.Crop
        }));

        // Create tensor with shape [1, 3, 224, 224] 4D
        // (batch_size, channels, height, width)
        var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });

        // Fill tensor with normalized pixel values
        for (int y = 0; y < ImageSize; y++)
        {
            for (int x = 0; x < ImageSize; x++)
            {
                var pixel = image[x, y];

                // Normalize each channel (R, G, B)
                tensor[0, 0, y, x] = (pixel.R / 255f - Mean[0]) / Std[0];  // Red
                tensor[0, 1, y, x] = (pixel.G / 255f - Mean[1]) / Std[1];  // Green
                tensor[0, 2, y, x] = (pixel.B / 255f - Mean[2]) / Std[2];  // Blue
            }
        }

        return tensor;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}