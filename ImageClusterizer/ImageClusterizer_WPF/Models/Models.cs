using CommunityToolkit.Mvvm.ComponentModel;
using System;
using System.Collections.Generic;

namespace ImageClusterizer.Models
{
    public enum VectorType
    {
        Embedding,
        Logit
    }

    public record ImageVector
    {
        public string FilePath { get; init; }
        public float[] Vector { get; init; }
        public VectorType VectorType { get; init; }
        public DateTime ProcessedAt { get; init; }
        public long FileSize { get; init; }
        public string? ThumbnailPath { get; init; }
        public float? PcaX { get; init; }
        public float? PcaY { get; init; }
    }

    public class ImageCluster
    {
        public int ClusterId { get; set; }
        public List<ImageVector> Images { get; set; } = new();
        public float[] Centroid { get; set; }
    }

    public record ScanProgress
    {
        public string CurrentFile { get; init; }
        public int ProcessedCount { get; init; }
        public int TotalCount { get; init; }
        public ImageVector? NewVector { get; init; }
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
        [ObservableProperty] private string thumbnailPath;
        /// <summary>File size in bytes used to compute dot radius on the map.</summary>
        [ObservableProperty] private long fileSize;
        /// <summary>
        /// Visual radius of the dot. Set by MainViewModel based on FileSize relative to dataset median.
        /// Range 6..22, default 14.
        /// </summary>
        [ObservableProperty] private double dotRadius = 14.0;
    }

    /// <summary>
    /// Lightweight dot data for the MapRenderCanvas high-performance renderer.
    /// Immutable - built once per render pass, swapped as a whole list.
    /// </summary>
    public class MapDot
    {
        public double X { get; init; }
        public double Y { get; init; }
        public double Radius { get; init; } = 14.0;
        public string FilePath { get; init; } = "";
        public string ThumbnailPath { get; init; } = "";
        public long FileSize { get; init; }
    }
}
