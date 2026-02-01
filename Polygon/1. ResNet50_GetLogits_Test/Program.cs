using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


if (!File.Exists(Constants.ModelPath))
{
    Console.WriteLine("Download ONNX Model Zoo from  https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx");
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
var inpits = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("data", input)
    //"data" is hardcoded inside ResNet
};

using var results = session.Run(inpits);

//get the logits (1000D vector) "resnetv24_dense0_fwd" hardcoded name SEE: Model Viewer https://netron.app/
//embedding  -> "resnetv24_pool1_fwd"
var logits = results.First(r => r.Name == Constants.OutputLayerName).AsEnumerable<float>();

Console.WriteLine(string.Join(" ", logits));

Console.ReadLine();

static class Constants
{
    public const string ModelPath = "resnet50-v2-7.onnx";
    public const int ImageSize = 224;
    public const string OutputLayerName = "resnetv24_dense0_fwd";
}

