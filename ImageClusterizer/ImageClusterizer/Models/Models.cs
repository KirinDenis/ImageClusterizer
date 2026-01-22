//TODO: separate files for each model
namespace ImageClusterizer.Models;

using System;
using System.Collections.Generic;

public record ImageVector
{
    public string FilePath { get; init; } // use init setter - set value just once 
    public float[] Vector { get; init; } // embedding от ResNet50
    public DateTime ProcessedAt { get; init; }
    public long FileSize { get; init; }
}

// Models/ImageCluster.cs
public class ImageCluster
{
    public int ClusterId { get; set; }
    public List<ImageVector> Images { get; set; } = new();
    public float[] Centroid { get; set; } // center of cluster
}

// Models/ScanProgress.cs
public record ScanProgress
{
    public string CurrentFile { get; init; }
    public int ProcessedCount { get; init; }
    public int TotalCount { get; init; }
    public ImageVector? NewVector { get; init; } // for UI refresh
}