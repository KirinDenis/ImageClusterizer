//SEE: 1. ResNet50_GetLogits_Test for first step

/// <summary>
/// ResNet50 Feature Extraction (Embedding Layer)
/// 
/// This application extracts 2048D feature embeddings from ResNe-50's penultimate layer.
/// 
/// IMPORTANT: This requires a modified ONNX model with layers cut after the embedding layer.
/// The standard resnet50-v2-7.onnx returns only 1000D logits (classification layer).
/// 
/// Setup Steps:
/// 1. Download the original model from ONNX Model Zoo:
///    https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
/// 
/// 2. Run the Python script to cut layers after the embedding:
///    python cut_to_embedded.py
///    
///    This script removes the final dense layer and sets "resnetv24_pool1_fwd" as the output,
///    producing a 2048D feature vector instead of 1000D logits.
/// 
/// 3. Use the modified model: resnet50-embedding-only.onnx
/// 
/// Processing Pipeline:
/// - Load image and resize to 224x224
/// - Normalize with ImageNet statistics (mean/std)
/// - Create 4D tensor [1, 3, 224, 224]
/// - Run inference through ResNet convolutional layers up to global average pooling
/// - Extract 2048D embedding from "resnetv24_pool1_fwd" layer
/// 
/// Use Cases:
/// - Image similarity search (cosine similarity between embeddings)
/// - Clustering and visualization (t-SNE, UMAP)
/// - Transfer learning
/// - Feature extraction for downstream tasks
/// 
/// For classification with 1000D logits, see: ResNet50_GetLogits_Test
/// 
/// Model viewer: https://netron.app/
/// </summary>

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


if (!File.Exists(Constants.ModelPath))
{
    Console.WriteLine("Download ONNX Model Zoo from  https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx");
    Console.WriteLine("after use cut_to_embedded.py to cut logits layer");
    return;
}

//Online model viewer 
//https://netron.app/
var session = new InferenceSession(Constants.ModelPath);

//some image from local assets
using var image = Image.Load<Rgb24>(@"Assets\cat.4001.jpg");

//resize to 224x224 for ResNet tensor compatible
image.Mutate(x => x.Resize(Constants.ImageSize, Constants.ImageSize));

//prepare 4D tensor  
// 1 is batch size (one image)
// 3 for RGB channels 
// 224 height 
// 224 width 
var input = new DenseTensor<float>(new[] { 1, 3, Constants.ImageSize, Constants.ImageSize });

//copy RGB BitMap pixels data to tensor 
image.ProcessPixelRows(accessor =>
{
    //for midlee image color values, from ResNet image statistics
    var mean = new[] { 0.485f, 0.456f, 0.406f };
    var std = new[] { 0.229f, 0.224f, 0.225f };

    //TODO: FFR: use Span here
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

//create data from tensor
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("data", input)
    //"data" is hardcoded inside ResNet
};

using var results = session.Run(inputs);

//get the embedding (2048D vector) ""resnetv24_pool1_fwd"" hardcoded name SEE: Model Viewer https://netron.app/
var embedding = results.First(r => r.Name == Constants.OutputLayerName).AsEnumerable<float>();

Console.WriteLine(string.Join(" ", embedding));

Console.ReadLine();

static class Constants
{
    //"cut_to_embedded.py to cut logits layer from resnet50-v2-7.onnx
    public const string ModelPath = "resnet50-embedding-only.onnx";
    public const int ImageSize = 224;
    public const string OutputLayerName = "resnetv24_pool1_fwd";
}

