//TODO: separate files for each model



using CommunityToolkit.Mvvm.ComponentModel;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Data;
using System;
using System.Collections.Generic;


namespace ImageClusterizer.Models
{ 

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
    [ObservableProperty]
    private int clusterId;

    [ObservableProperty]
    private double x;

    [ObservableProperty]
    private double y;

    [ObservableProperty]
    private int imageCount;

    public string Label = ""; // => $"Cluster {ClusterId} ({ImageCount} images)";
}

public partial class ImageVisualItem : ObservableObject
{
    [ObservableProperty]
    private int clusterId;

    [ObservableProperty]
    private double x;

    [ObservableProperty]
    private double y;

    [ObservableProperty]
    private string filePath;

    [ObservableProperty]
    private string thumbnailPath;
}
}

namespace ImageClusterizer.Converters
{
    public class PositionToMarginConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, string language)
        {
            if (value is ImageClusterizer.Models.ImageVisualItem item)
            {
                return new Thickness(item.X, item.Y, 0, 0);
            }
            return new Thickness(0);
        }

        public object ConvertBack(object value, Type targetType, object parameter, string language)
        {
            throw new NotImplementedException();
        }
    }
}


