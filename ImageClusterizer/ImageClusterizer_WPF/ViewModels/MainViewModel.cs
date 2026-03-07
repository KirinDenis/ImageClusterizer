namespace ImageClusterizer.ViewModels;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Models;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

public partial class MainViewModel : ObservableObject
{
    private readonly ImageScanner imageScanner;
    private readonly IVectorDatabase vectorDatabase;
    private readonly ClusteringService clusteringService;
    private readonly StorageService storageService;

    // --- Cluster and image collections ---
    [ObservableProperty]
    private ObservableCollection<ImageCluster> clusters = new();

    // --- Scan progress state ---
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

    // --- Vector type selection ---
    [ObservableProperty]
    private VectorType selectedVectorType = VectorType.Embedding;

    public IReadOnlyList<VectorType> AvailableVectorTypes { get; } =
        Enum.GetValues<VectorType>().ToList();

    // --- Canvas visualization ---
    [ObservableProperty]
    private ObservableCollection<ClusterVisualItem> clusterItems = new();

    [ObservableProperty]
    private ObservableCollection<ImageVisualItem> imageItems = new();

    [ObservableProperty]
    private double canvasWidth = 1000;

    [ObservableProperty]
    private double canvasHeight = 1000;

    private CancellationTokenSource? cts;

    public MainViewModel(
        ImageScanner imageScanner,
        IVectorDatabase vectorDatabase,
        ClusteringService clusteringService,
        StorageService storageService)
    {
        this.imageScanner     = imageScanner;
        this.vectorDatabase   = vectorDatabase;
        this.clusteringService = clusteringService;
        this.storageService   = storageService;
    }

    // ---- Scan command ----

    [RelayCommand]
    private async Task StartScanImagesAsync()
    {
        string? folder = Utility.SelectFolderDiagoAsync();
        if (string.IsNullOrWhiteSpace(folder)) return;

        IsScanning = true;
        cts = new CancellationTokenSource();

        try
        {
            Clusters.Clear();

            await foreach (var progress in imageScanner.ScanFolderAsync(folder, SelectedVectorType, cts.Token))
            {
                CurrentFile    = Path.GetFileName(progress.CurrentFile);
                ProcessedCount = progress.ProcessedCount;
                TotalCount     = progress.TotalCount;
                Progress       = (double)ProcessedCount / TotalCount * 100;
            }

            await LoadAndDisplayAsync();
        }
        finally
        {
            IsScanning = false;
            cts?.Cancel();
            cts?.Dispose();
        }
    }

    // ---- Reload from database ----

    [RelayCommand]
    private async Task LoadExistingClustersAsync()
    {
        await LoadAndDisplayAsync();
    }

    // ---- Cancel scan ----

    [RelayCommand]
    private void CancelScan()
    {
        cts?.Cancel();
    }

    // ---- Clear all data ----

    [RelayCommand]
    private async Task ClearAllDataAsync()
    {
        var result = MessageBox.Show(
            "This will permanently delete all stored vectors, thumbnails, and cached positions." +
            "\n\nYour original image files will NOT be affected." +
            "\n\nContinue?",
            "Clear all data",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning);

        if (result != MessageBoxResult.Yes) return;

        await storageService.ClearAllDataAsync();

        // Clear all UI collections
        Clusters.Clear();
        ClusterItems.Clear();
        ImageItems.Clear();
        CurrentFile    = "";
        ProcessedCount = 0;
        TotalCount     = 0;
        Progress       = 0;
    }

    // ---- Core logic ----

    /// <summary>
    /// Loads all vectors from DB. If PCA coordinates are fully cached, renders scatter immediately.
    /// If any vector is missing PCA coordinates, recomputes PCA for all and saves results.
    /// </summary>
    private async Task LoadAndDisplayAsync()
    {
        var vectors = await vectorDatabase.GetAllAsync();
        if (vectors.Count == 0) return;

        bool pcaCacheComplete = vectors.All(v => v.PcaX.HasValue && v.PcaY.HasValue);

        if (pcaCacheComplete)
        {
            // Fast path: use cached PCA coordinates, skip expensive SVD computation
            PopulateImageItemsFromCache(vectors);
        }
        else
        {
            // Slow path: compute PCA, save coordinates, then display
            await ComputeAndCachePcaAsync(vectors);
        }

        // Also update cluster collection for Clusters tab (lazy — only structure, no computation)
        await ClusterImagesAsync(vectors);
    }

    /// <summary>
    /// Populates ImageItems directly from cached PCA coordinates in the DB.
    /// No recomputation needed.
    /// </summary>
    private void PopulateImageItemsFromCache(List<ImageVector> vectors)
    {
        ImageItems.Clear();
        ClusterItems.Clear();

        // Normalize cached coordinates to canvas size
        var minX = vectors.Min(v => v.PcaX!.Value);
        var maxX = vectors.Max(v => v.PcaX!.Value);
        var minY = vectors.Min(v => v.PcaY!.Value);
        var maxY = vectors.Max(v => v.PcaY!.Value);

        double rangeX = Math.Max(maxX - minX, 0.0001);
        double rangeY = Math.Max(maxY - minY, 0.0001);
        double padding = 0.05;
        double usableW = CanvasWidth  * (1 - 2 * padding);
        double usableH = CanvasHeight * (1 - 2 * padding);

        foreach (var v in vectors)
        {
            ImageItems.Add(new ImageVisualItem
            {
                FilePath      = v.FilePath,
                ThumbnailPath = v.ThumbnailPath ?? v.FilePath, // fallback to original if no thumbnail
                X             = (v.PcaX!.Value - minX) / rangeX * usableW + CanvasWidth  * padding,
                Y             = (v.PcaY!.Value - minY) / rangeY * usableH + CanvasHeight * padding
            });
        }
    }

    /// <summary>
    /// Runs PCA computation on background thread, saves coordinates to DB, then displays results.
    /// </summary>
    private async Task ComputeAndCachePcaAsync(List<ImageVector> vectors)
    {
        var positions = await Task.Run(() =>
            clusteringService.CalculatePositions(
                new List<ImageCluster> { new ImageCluster { Images = vectors, ClusterId = 0 } },
                (int)CanvasWidth,
                (int)CanvasHeight));

        ImageItems.Clear();
        ClusterItems.Clear();

        // Save PCA coordinates to DB and populate UI
        var saveTasks = new List<Task>();

        foreach (var pos in positions.Where(p => !p.IsCentroid))
        {
            ImageItems.Add(new ImageVisualItem
            {
                FilePath      = pos.ImageVector.FilePath,
                ThumbnailPath = pos.ImageVector.ThumbnailPath ?? pos.ImageVector.FilePath,
                X             = pos.X,
                Y             = pos.Y
            });

            // Persist PCA coordinates for next startup (fire-and-forget batch)
            saveTasks.Add(vectorDatabase.SavePcaCoordinatesAsync(
                pos.ImageVector.FilePath,
                (float)pos.X,
                (float)pos.Y));
        }

        // Save all PCA coordinates in parallel
        await Task.WhenAll(saveTasks);
    }

    /// <summary>Computes cosine similarity clusters and updates the Clusters collection</summary>
    private async Task ClusterImagesAsync(List<ImageVector> vectors)
    {
        var clusterList = await Task.Run(() => clusteringService.ClusterBySimilarity(vectors, 0.5f));

        Clusters.Clear();
        foreach (var cluster in clusterList)
        {
            Clusters.Add(cluster);
        }
    }

    /// <summary>Builds cluster visual items from an existing cluster list for the scatter view</summary>
    public void LoadClusters(List<ImageCluster> clusters)
    {
        ClusterItems.Clear();

        var positions = clusteringService.CalculatePositions(clusters, (int)CanvasWidth, (int)CanvasHeight);
        var grouped   = positions.GroupBy(p => p.ClusterId);

        foreach (var group in grouped)
        {
            var centroid = group.FirstOrDefault(p => p.IsCentroid);
            if (centroid != null)
            {
                ClusterItems.Add(new ClusterVisualItem
                {
                    ClusterId  = centroid.ClusterId,
                    X          = centroid.X,
                    Y          = centroid.Y,
                    ImageCount = group.Count(p => !p.IsCentroid)
                });
            }
        }
    }
}
