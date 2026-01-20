using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

var session = new InferenceSession("resnet50-v2-7.onnx");

using var image = Image.Load<Rgb24>(@"E:\Downloads\WhatsApp Image 2026-01-14 at 2.18.22 PM.jpeg");
image.Mutate(x => x.Resize(224, 224));

var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

image.ProcessPixelRows(accessor =>
{
    //for midlee image color values, from ResNet image statistics
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

var inpits = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("data", input)
};

using var results = session.Run(inpits);

var embedding = results.First(r => r.Name == "resnetv24_dense0_fwd").AsEnumerable<float>().ToArray();


Console.ReadLine();
