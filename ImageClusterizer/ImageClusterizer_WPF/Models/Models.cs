using CommunityToolkit.Mvvm.ComponentModel;
using System;
using System.Collections.Generic;

namespace ImageClusterizer.Models
{
    /// <summary>
    /// Specifies the type of vector extracted from the neural network
    /// </summary>
    public enum VectorType
    {
        /// <summary>2048-dimensional embedding from penultimate ResNet layer (better for similarity)</summary>
        Embedding,

        /// <summary>1000-dimensional logit output from final ResNet classification layer</summary>
        Logit
    }

    public record ImageVector
    {
        public string FilePath { get; init; }         // original image path (not stored in thumbnail path)
        public float[] Vector { get; init; }          // embedding or logit vector from ResNet50
        public VectorType VectorType { get; init; }   // type of vector stored
        public DateTime ProcessedAt { get; init; }
        public long FileSize { get; init; }

        // Thumbnail cache: path to 224x224 JPEG saved during scan
        public string? ThumbnailPath { get; init; }

        // PCA 2D position cache: null = not yet computed
        public float? PcaX { get; init; }
        public float? PcaY { get; init; }
    }

    public class ImageCluster
    {
        public int ClusterId { get; set; }
        public List<ImageVector> Images { get; set; } = new();
        public float[] Centroid { get; set; } // center of cluster
    }

    public record ScanProgress
    {
        public string CurrentFile { get; init; }
        public int ProcessedCount { get; init; }
        public int TotalCount { get; init; }
        public ImageVector? NewVector { get; init; } // for UI refresh
    }

    public class VectorInfo
    {
        public int ClusterId { get; set; }
        public bool IsCentroid { get; set; }
        public ImageVector ImageVector { get; set; }
    }

    public class ClusterPosition
    {
        public int ClusterId { get; set; }
        public bool IsCentroid { get; set; }
        public ImageVector ImageVector { get; set; }
        public double X { get; set; }
        public double Y { get; set; }
    }

    public partial class ClusterVisualItem : ObservableObject
    {
        [ObservableProperty] private int clusterId;
        [ObservableProperty] private double x;
        [ObservableProperty] private double y;
        [ObservableProperty] private int imageCount;
        public string Label = "";
    }

    public partial class ImageVisualItem : ObservableObject
    {
        [ObservableProperty] private int clusterId;
        [ObservableProperty] private double x;
        [ObservableProperty] private double y;
        [ObservableProperty] private string filePath;
        [ObservableProperty] private string thumbnailPath; // points to cached thumbnail, not original
    }
}
