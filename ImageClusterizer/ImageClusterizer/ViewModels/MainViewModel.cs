namespace ImageClusterizer.ViewModels;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Models;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using Microsoft.UI.Xaml;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Windows.Storage;


public partial class MainViewModel : ObservableObject
{
    private readonly ImageScanner imageScanner;
    private readonly IVectorDatabase vectorDatabase;
    private readonly ClusteringService clusteringService;


    [ObservableProperty]
    private ObservableCollection<ImageCluster> clusters = new();

    [ObservableProperty]
    private string currentFile = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProcessedCountText))]
    private int processedCount;

    public string ProcessedCountText => ProcessedCount.ToString();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(TotalCountText))]
    private int totalCount;

    public string TotalCountText => TotalCount.ToString();

    [ObservableProperty]
    private double progress;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsNotScanning), nameof(ProgressVisibility))]
    private bool isScanning;

    public bool IsNotScanning => !IsScanning;
    public Visibility ProgressVisibility => IsScanning ? Visibility.Visible : Visibility.Collapsed;


    [ObservableProperty]
    private ObservableCollection<ClusterVisualItem> clusterItems = new();

    [ObservableProperty]
    private ObservableCollection<ImageVisualItem> imageItems = new();

    [ObservableProperty]
    private double canvasWidth = 1000;

    [ObservableProperty]
    private double canvasHeight = 1000;


    private CancellationTokenSource? cts;
    public MainViewModel(ImageScanner imageScanner, IVectorDatabase vectorDatabase, ClusteringService clusteringService)
    {
        this.imageScanner = imageScanner;
        this.vectorDatabase = vectorDatabase;
        this.clusteringService = clusteringService;
    }

    [RelayCommand]
    private async Task StartScanImagesAsync()
    {
        StorageFolder folder = await Utility.SelectFolderDiagoAsync();

        if (folder == null)
        {
            return;
        }

        IsScanning = true;
        cts = new CancellationTokenSource();

        try
        {
            Clusters.Clear();

            await foreach (var progress in imageScanner.ScanFolderAsync(folder.Path, cts.Token))
            {
                CurrentFile = Path.GetFileName(progress.CurrentFile);
                ProcessedCount = progress.ProcessedCount;
                TotalCount = progress.TotalCount;
                Progress = (double)ProcessedCount / TotalCount * 100;
            }

            await ClusterImagesAsync();
        }
        finally
        {
            IsScanning = false;
            cts?.Cancel();
            cts?.Dispose();
        }
    }

    public void LoadClusters(List<ImageCluster> clusters)
    {
        ClusterItems.Clear();
        ImageItems.Clear();

        var positions = clusteringService.CalculatePositions(clusters,
            (int)CanvasWidth, (int)CanvasHeight);

        var grouped = positions.GroupBy(p => p.ClusterId);

        foreach (var group in grouped)
        {

            var centroid = group.FirstOrDefault(p => p.IsCentroid);
            if (centroid != null)
            {
                ClusterItems.Add(new ClusterVisualItem
                {
                    ClusterId = centroid.ClusterId,
                    X = centroid.X,
                    Y = centroid.Y,
                    ImageCount = group.Count(p => !p.IsCentroid)
                });
            }

            foreach (var imagePos in group.Where(p => !p.IsCentroid))
            {
                ImageItems.Add(new ImageVisualItem
                {
                    ClusterId = imagePos.ClusterId,
                    X = imagePos.X,
                    Y = imagePos.Y,
                    FilePath = imagePos.ImageVector.FilePath,
                    ThumbnailPath = imagePos.ImageVector.FilePath
                });
            }
        }
    }


    [RelayCommand]
    private async Task LoadExistingClustersAsync()
    {
        await ClusterImagesAsync();
        LoadClusters(Clusters.ToList());
    }

    [RelayCommand]
    private void CancelScan()
    {
        cts?.Cancel();
    }


    private async Task ClusterImagesAsync()
    {
        var vectors = await vectorDatabase.GetAllAsync();
        var clusters = await Task.Run(() => clusteringService.ClusterBySimilarity(vectors, 0.5f));

        Clusters.Clear();
        foreach (var cluster in clusters)
        {
            Clusters.Add(cluster);
        }
    }
}

