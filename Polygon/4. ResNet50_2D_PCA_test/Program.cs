//SEE: 1. ResNet50_GetLogits_Test
//SEE: 2. ResNet50_GetEmbedding_Test
//SEE: 3.ResNet50_Image_similarity_search_test
//BEFORE this step

// 2D Principal Component Analysis (PCA) based on ResNet50 embeddings vectors

//SEE: Reduce2DTo1D_PCA method for details - how PCA works (2D vectors to 1D projection)

namespace ResNet50_Image_similarity_search_test;

using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

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

    /// <summary>
    /// Normalize PCA 2D position 
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <returns></returns>
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

    /// <summary>
    /// SEE: Reduce2DTo1D_PCA for details
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
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

        for (int i = 0; i < vectors_count; i++)
        {
            for (int j = 0; j < one_vector_dimension; j++)
            {
                matrix[i, j] -= columnMeans[j];
            }
        }

        // SVD
        //Singulat value decomposition 
        var svd = matrix.Svd(computeVectors: true);
        var u = svd.U; //left singular vectors, normalazed to -1..1
        var s = svd.S; //singular values, matrix diagonal 


        var result = new double[vectors_count][];
        for (int i = 0; i < vectors_count; i++)
        {
            result[i] = new double[]
            {
                    u[i, 0] * s[0], //vector projection to first axis - X, s[0] a large variance 
                    u[i, 1] * s[1]  //vector projection to second axis - Y, s[1] a large variance but smaller then s[0]
            };
        }

        return result;
    }


    /// <summary>
    /// AI (Sonnet) generated comments - how to calculate PCA -> SVD manualy 
    /// --------------------------------------------------------------------
    /// 
    /// Reduces 2D vectors to 1D using PCA (Principal Component Analysis) via SVD.
    ///
    /// MATH BEHIND (step by step example with 4 points):
    ///
    /// Input points: (1,2), (2,3), (3,4), (4,5)
    ///
    /// STEP 1 — Centralization (shift center of mass to origin 0,0):
    ///   mean = (2.5, 3.5)
    ///   centered points:
    ///     (-1.5, -1.5), (-0.5, -0.5), (0.5, 0.5), (1.5, 1.5)
    ///
    /// STEP 2 — Covariance matrix AᵀA (2x2):
    ///   shows how much X and Y vary together (off-diagonal)
    ///   and individual variance per axis (diagonal)
    ///
    ///   AᵀA = Aᵀ * A =
    ///   (-1.5*-1.5 + -0.5*-0.5 + 0.5*0.5 + 1.5*1.5) = 2.25+0.25+0.25+2.25 = 5.0
    ///
    ///   AᵀA = [5.0  5.0]   <- X variance=5, cov(X,Y)=5 (move together perfectly)
    ///         [5.0  5.0]   <- cov(X,Y)=5, Y variance=5
    ///
    /// STEP 3 — Eigenvalues λ of AᵀA (how much variance along each axis direction):
    ///   det(AᵀA - λI) = 0
    ///   (5-λ)² - 25 = 0
    ///   (-λ)(10-λ) = 0
    ///   λ1 = 10  <- large, this axis captures all the variance
    ///   λ2 = 0   <- zero, no variance in perpendicular direction (points on perfect line)
    ///
    /// STEP 4 — Eigenvectors V (principal directions / axes of max spread):
    ///   for λ1=10:  (AᵀA - 10I) * v = 0  ->  vx = vy  ->  v1 = [1, 1]
    ///   normalize:  length = √(1²+1²) = √2 = 1.414
    ///   v1 = [1/1.414, 1/1.414] = [0.707, 0.707]  <- 45° diagonal, axis of max spread
    ///   v2 = [-0.707, 0.707]                        <- perpendicular, no spread
    ///
    /// STEP 5 — Singular value S (scale factor along principal axis):
    ///   S = √λ1 = √10 ≈ 3.16  <- L2Norm of matrix = largest singular value
    ///
    /// STEP 6 — Project all points onto principal axis (1D coordinates):
    ///   projection = dot(point, v1) = x*0.707 + y*0.707
    ///     (-1.5,-1.5) -> -2.12
    ///     (-0.5,-0.5) -> -0.707
    ///     ( 0.5, 0.5) ->  0.707
    ///     ( 1.5, 1.5) ->  2.12
    ///
    ///   same result via SVD: U[i,0] * S
    ///   because mathematically: A * V = U * S
    ///   U stores normalized projections (-1..1), multiply by S to restore real scale
    ///
    /// WHY SVD instead of manual:
    ///   SVD computes U, S, V simultaneously, numerically stable, works for any dimension.
    ///   U[:,0] * S[0] == A * V[:,0]  (first column = projection onto axis of max variance)
    /// </summary>

    private static double[] Reduce2DTo1D_PCA(double[,] vectors2D)
    {

        var matrix = Matrix<double>.Build.DenseOfArray(vectors2D);

        int dimension = 2;
        int matrixRowSize = vectors2D.Length / dimension;

        //Centalize vectors
        var columnMeans = matrix.ColumnSums() / (matrixRowSize);

        for (int i = 0; i < matrixRowSize; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                matrix[i, j] -= columnMeans[j];
            }
        }

        // SVD
        //Singulat Value Decomposition 
        var svd = matrix.Svd(computeVectors: true);
        var u = svd.U; //left singular vectors, ready calculated A * V (matrix * |0.707 0.707| vector)
        double s = svd.L2Norm; // or use -> = svd.S.Max() -> take the value along the axis with a large variance

        var result = new double[matrixRowSize];
        for (int i = 0; i < matrixRowSize; i++)
        {
            result[i] = u[i, 0] * s; // projection to 1D axis
        }
        return result;
    }

    static void Main(string[] args)
    {
        Console.Clear();

        //PCA -> SVD test 
        //2D vectors to 1D PCA
        double[,] vectors2D_1 = { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 } };
        Console.WriteLine($"2D to 1D test 1 = {string.Join(',', Reduce2DTo1D_PCA(vectors2D_1))}");

        double[,] vectors2D_2 = { { 2, 2 }, { 2, 2.2 }, { 2, 3 }, { 4, -5 } };
        Console.WriteLine($"2D to 1D test 1 = {string.Join(',', Reduce2DTo1D_PCA(vectors2D_2))}");

        //PCS for ebeddings section 
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

        Console.ForegroundColor = ConsoleColor.White;
        Console.BackgroundColor = ConsoleColor.Blue;

        for (int i = 0; i < normalized.Length; i++)
        {
            Console.CursorLeft = (int)normalized[i][0];
            Console.CursorTop = (int)normalized[i][1];
            Console.WriteLine(Path.GetFileName(embeddings[i].imageFile));
        }

        Console.ReadLine();
    }
}
