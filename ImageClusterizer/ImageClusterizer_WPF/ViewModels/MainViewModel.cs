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
    private readonly ThemeService themeService;
    private readonly LogService logService;

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
    public IReadOnlyList<VectorType> AvailableVectorTypes { get; } = Enum.GetValues<VectorType>().ToList();

    // --- Canvas visualization ---
    [ObservableProperty]
    private ObservableCollection<ClusterVisualItem> clusterItems = new();

    [ObservableProperty]
    private ObservableCollection<ImageVisualItem> imageItems = new();

    [ObservableProperty]
    private double canvasWidth = 1000;

    [ObservableProperty]
    private double canvasHeight = 1000;

    // --- PCA state ---
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RecalculatePcaCommand))]
    private bool isPcaComputing;

    [ObservableProperty]
    private int pcaProgress;

    // --- Clustering state ---
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))]
    private bool isClusterComputing;

    [ObservableProperty]
    private float similarityThreshold = 0.85f;

    // --- Advanced settings ---
    [ObservableProperty]
    private int sparseTopN = 2048;

    [ObservableProperty]
    private bool useGpu = true;

    [ObservableProperty]
    private int threadCount = 0;   // 0 = auto (Environment.ProcessorCount)

    // --- GPU info ---
    [ObservableProperty]
    private bool gpuAvailable;

    [ObservableProperty]
    private string gpuName = "Detecting...";

    // --- Live telemetry ---
    [ObservableProperty]
    private int vectorCount;

    [ObservableProperty]
    private int clusterCount;

    [ObservableProperty]
    private string databaseSizeText = "0 KB";

    [ObservableProperty]
    private string lastScanDuration = "-";

    // --- Console log ---
    [ObservableProperty]
    private ObservableCollection<string> consoleLines = new();

    [ObservableProperty]
    private bool isConsoleExpanded = true;

    // --- Cockpit panel ---
    [ObservableProperty]
    private bool isCockpitExpanded = false;

    // --- Theme ---
    [ObservableProperty]
    private string themeIcon = "Light";

    private CancellationTokenSource? cts;

    public MainViewModel(
        ImageScanner imageScanner,
        IVectorDatabase vectorDatabase,
        ClusteringService clusteringService,
        StorageService storageService,
        ThemeService themeService,
        LogService logService)
    {
        this.imageScanner = imageScanner;
        this.vectorDatabase = vectorDatabase;
        this.clusteringService = clusteringService;
        this.storageService = storageService;
        this.themeService = themeService;
        this.logService = logService;

        // Subscribe to log service — dispatch to UI thread, trim to MaxLines
        logService.LogAdded += line =>
        {
            Application.Current?.Dispatcher.InvokeAsync(() =>
            {
                ConsoleLines.Add(line);
                while (ConsoleLines.Count > LogService.MaxLines)
                    ConsoleLines.RemoveAt(0);
            });
        };

        // Detect GPU on background thread so startup is not blocked
        Task.Run(() =>
        {
            var info = GpuDetector.Detect();
            Application.Current?.Dispatcher.InvokeAsync(() =>
            {
                GpuAvailable = info.IsAvailable;
                GpuName = info.DeviceName;
                logService.Log($"GPU detection: {info.DeviceName} ({info.ProviderName})");
            });
        });

        // Load settings
        var settings = AppSettings.Load();
        SparseTopN = settings.SparseTopN;
        UseGpu = settings.UseGpu;
        ThreadCount = settings.ThreadCount;
        SimilarityThreshold = (float)settings.SimilarityThreshold;
        IsConsoleExpanded = settings.IsConsoleExpanded;

        UpdateThemeIcon();
        logService.Log("ImageClusterizer started.");
        UpdateDatabaseSize();
    }

    // ---- Theme ----
    [RelayCommand]
    private void ToggleTheme()
    {
        themeService.ToggleTheme();
        UpdateThemeIcon();
        logService.Log($"Theme switched to: {themeService.CurrentTheme}");
    }

    private void UpdateThemeIcon()
    {
        ThemeIcon = themeService.CurrentTheme == ThemeService.Theme.Dark ? "Dark" : "Light";
    }

    // ---- Console ----
    [RelayCommand]
    private void ClearConsole()
    {
        ConsoleLines.Clear();
        logService.Clear();
    }

    // ---- Advanced settings save ----
    partial void OnSparseTopNChanged(int value)
    {
        logService.Log($"Vector compression changed: Top-{value}/2048");
        SaveSettings();
    }

    partial void OnSimilarityThresholdChanged(float value)
    {
        logService.Log($"Similarity threshold changed: {value:F2}");
        SaveSettings();
    }

    partial void OnUseGpuChanged(bool value)
    {
        logService.Log($"Use GPU: {value}");
        SaveSettings();
    }

    partial void OnIsConsoleExpandedChanged(bool value) => SaveSettings();

    private void SaveSettings()
    {
        var settings = AppSettings.Load();
        settings.SparseTopN = SparseTopN;
        settings.UseGpu = UseGpu;
        settings.ThreadCount = ThreadCount;
        settings.SimilarityThreshold = SimilarityThreshold;
        settings.IsConsoleExpanded = IsConsoleExpanded;
        AppSettings.Save(settings);
    }

    // ---- Scan command ----
    [RelayCommand]
    private async Task StartScanImagesAsync()
    {
        string? folder = Utility.SelectFolderDiagoAsync();
        if (string.IsNullOrWhiteSpace(folder)) return;

        IsScanning = true;
        cts = new CancellationTokenSource();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            Clusters.Clear();
            logService.Log($"Scanning folder: {folder}");

            await foreach (var prog in imageScanner.ScanFolderAsync(folder, SelectedVectorType, cts.Token))
            {
                CurrentFile = Path.GetFileName(prog.CurrentFile);
                ProcessedCount = prog.ProcessedCount;
                TotalCount = prog.TotalCount;
                Progress = (double)ProcessedCount / TotalCount * 100;

                if (ProcessedCount % 50 == 0)
                    logService.Log($"Vectorizing: {ProcessedCount}/{TotalCount} — {CurrentFile}");
            }

            sw.Stop();
            LastScanDuration = $"{sw.Elapsed.TotalSeconds:F1}s";
            logService.Log($"Scan complete — {ProcessedCount} vectors in {LastScanDuration}");

            await LoadAndDisplayAsync();
        }
        finally
        {
            IsScanning = false;
            cts?.Cancel();
            cts?.Dispose();
            UpdateDatabaseSize();
        }
    }

    // ---- Reload from database ----
    [RelayCommand]
    private async Task LoadExistingClustersAsync()
    {
        logService.Log("Loading existing vectors from database...");
        await LoadAndDisplayAsync();
    }

    // ---- Cancel scan ----
    [RelayCommand]
    private void CancelScan()
    {
        cts?.Cancel();
        logService.Log("Scan cancelled by user.");
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

        logService.Log("Clearing all data...");
        await storageService.ClearAllDataAsync();

        Clusters.Clear();
        ClusterItems.Clear();
        ImageItems.Clear();
        CurrentFile = "";
        ProcessedCount = 0;
        TotalCount = 0;
        Progress = 0;
        VectorCount = 0;
        ClusterCount = 0;
        UpdateDatabaseSize();
        logService.Log("All data cleared.");
    }

    // ---- Recalculate PCA ----
    [RelayCommand(CanExecute = nameof(CanRecalculatePca))]
    private async Task RecalculatePcaAsync()
    {
        logService.Log("Recalculate PCA requested — clearing cache...");
        await vectorDatabase.ClearPcaCacheAsync();
        var vectors = await vectorDatabase.GetAllAsync();
        if (vectors.Count == 0)
        {
            logService.Log("No vectors in database — nothing to recalculate.");
            return;
        }
        await ComputeAndCachePcaAsync(vectors);
    }

    private bool CanRecalculatePca() => !IsPcaComputing && !IsScanning;

    // ---- Compute clusters (lazy) ----
    [RelayCommand(CanExecute = nameof(CanComputeClusters))]
    private async Task ComputeClustersAsync()
    {
        IsClusterComputing = true;
        logService.Log($"Starting cosine similarity clustering (threshold: {SimilarityThreshold:F2})...");

        try
        {
            var vectors = await vectorDatabase.GetAllAsync();
            var clusterList = await Task.Run(() =>
                clusteringService.ClusterBySimilarity(vectors, SimilarityThreshold));

            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                Clusters.Clear();
                foreach (var c in clusterList)
                    Clusters.Add(c);
                ClusterCount = Clusters.Count;
            });

            logService.Log($"Clustering complete — {ClusterCount} clusters from {vectors.Count} images.");
        }
        finally
        {
            IsClusterComputing = false;
        }
    }

    private bool CanComputeClusters() => !IsClusterComputing && !IsScanning && !IsPcaComputing;

    // ---- Core logic ----

    private async Task LoadAndDisplayAsync()
    {
        var vectors = await vectorDatabase.GetAllAsync();
        VectorCount = vectors.Count;

        if (vectors.Count == 0)
        {
            logService.Log("Database is empty — scan a folder first.");
            return;
        }

        logService.Log($"Loaded {vectors.Count} vectors from database.");

        bool pcaCacheComplete = vectors.All(v => v.PcaX.HasValue && v.PcaY.HasValue);

        if (pcaCacheComplete)
        {
            logService.Log("PCA cache is complete — rendering from cache (fast path).");
            await PopulateImageItemsFromCacheAsync(vectors);
        }
        else
        {
            logService.Log($"PCA cache incomplete ({vectors.Count(v => !v.PcaX.HasValue)} missing) — computing SVD...");
            await ComputeAndCachePcaAsync(vectors);
        }

        UpdateDatabaseSize();
    }

    private async Task PopulateImageItemsFromCacheAsync(List<ImageVector> vectors)
    {
        var minX = vectors.Min(v => v.PcaX!.Value);
        var maxX = vectors.Max(v => v.PcaX!.Value);
        var minY = vectors.Min(v => v.PcaY!.Value);
        var maxY = vectors.Max(v => v.PcaY!.Value);
        double rangeX = Math.Max(maxX - minX, 0.0001);
        double rangeY = Math.Max(maxY - minY, 0.0001);
        double padding = 0.05;
        double usableW = CanvasWidth  * (1 - 2 * padding);
        double usableH = CanvasHeight * (1 - 2 * padding);

        ImageItems.Clear();
        ClusterItems.Clear();

        const int batchSize = 100;
        int batchNum = 0;
        int totalBatches = (vectors.Count + batchSize - 1) / batchSize;

        foreach (var chunk in vectors.Chunk(batchSize))
        {
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                foreach (var v in chunk)
                {
                    ImageItems.Add(new ImageVisualItem
                    {
                        FilePath      = v.FilePath,
                        ThumbnailPath = v.ThumbnailPath ?? v.FilePath,
                        X = (v.PcaX!.Value - minX) / rangeX * usableW + CanvasWidth  * padding,
                        Y = (v.PcaY!.Value - minY) / rangeY * usableH + CanvasHeight * padding
                    });
                }
            });
            batchNum++;
            if (batchNum % 5 == 0)
                logService.Log($"Rendering batch {batchNum}/{totalBatches}");
            await Task.Delay(1);
        }

        logService.Log($"Map rendered — {ImageItems.Count} images.");
    }

    private async Task ComputeAndCachePcaAsync(List<ImageVector> vectors)
    {
        IsPcaComputing = true;
        PcaProgress = 0;

        int dim = vectors.Count > 0 ? vectors[0].Vector.Length : 0;
        logService.Log($"PCA: SVD on {vectors.Count} x {dim} matrix (Top-{SparseTopN})...");

        var prog = new Progress<(int current, int total, string message)>(p =>
        {
            PcaProgress = p.total > 0 ? p.current * 100 / p.total : 0;
            if (!string.IsNullOrEmpty(p.message))
                logService.Log(p.message);
        });

        List<ClusterPosition> positions;
        try
        {
            positions = await Task.Run(() =>
                clusteringService.CalculatePositionsSparse(
                    new List<ImageCluster> { new ImageCluster { Images = vectors, ClusterId = 0 } },
                    (int)CanvasWidth, (int)CanvasHeight,
                    SparseTopN, prog));
        }
        finally
        {
            IsPcaComputing = false;
        }

        logService.Log($"PCA complete — rendering {positions.Count} points...");

        ImageItems.Clear();
        ClusterItems.Clear();

        const int batchSize = 100;
        var nonCentroids = positions.Where(p => !p.IsCentroid).ToList();
        int batchNum = 0;
        int totalBatches = (nonCentroids.Count + batchSize - 1) / batchSize;
        var saveTasks = new List<Task>();

        foreach (var chunk in nonCentroids.Chunk(batchSize))
        {
            var chunkList = chunk.ToList();

            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                foreach (var pos in chunkList)
                {
                    ImageItems.Add(new ImageVisualItem
                    {
                        FilePath      = pos.ImageVector.FilePath,
                        ThumbnailPath = pos.ImageVector.ThumbnailPath ?? pos.ImageVector.FilePath,
                        X = pos.X,
                        Y = pos.Y
                    });
                }
            });

            batchNum++;
            if (batchNum % 5 == 0)
                logService.Log($"Rendering batch {batchNum}/{totalBatches}");
            await Task.Delay(1);

            foreach (var pos in chunkList)
            {
                saveTasks.Add(vectorDatabase.SavePcaCoordinatesAsync(
                    pos.ImageVector.FilePath, (float)pos.X, (float)pos.Y));
            }
        }

        await Task.WhenAll(saveTasks);
        logService.Log($"Map fully rendered — {ImageItems.Count} images. PCA coordinates saved.");
    }

    private void UpdateDatabaseSize()
    {
        try
        {
            var dbPath = storageService.DatabasePath;
            if (File.Exists(dbPath))
            {
                var bytes = new FileInfo(dbPath).Length;
                DatabaseSizeText = bytes < 1_048_576
                    ? $"{bytes / 1024.0:F1} KB"
                    : $"{bytes / 1_048_576.0:F1} MB";
            }
            else
            {
                DatabaseSizeText = "0 KB";
            }
        }
        catch
        {
            DatabaseSizeText = "N/A";
        }
    }
}
