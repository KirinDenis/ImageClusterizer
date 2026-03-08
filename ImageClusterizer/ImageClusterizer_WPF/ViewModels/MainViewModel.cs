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

    [ObservableProperty] private ObservableCollection<ImageCluster> clusters = new();
    [ObservableProperty] private string currentFile = "";
    [ObservableProperty][NotifyPropertyChangedFor(nameof(ProcessedCountText))] private int processedCount;
    public string ProcessedCountText => ProcessedCount.ToString();
    [ObservableProperty][NotifyPropertyChangedFor(nameof(TotalCountText))] private int totalCount;
    public string TotalCountText => TotalCount.ToString();
    [ObservableProperty] private double progress;
    [ObservableProperty][NotifyPropertyChangedFor(nameof(IsNotScanning), nameof(ProgressVisibility))] private bool isScanning;
    public bool IsNotScanning => !IsScanning;
    public Visibility ProgressVisibility => IsScanning ? Visibility.Visible : Visibility.Collapsed;
    [ObservableProperty] private VectorType selectedVectorType = VectorType.Embedding;
    public IReadOnlyList<VectorType> AvailableVectorTypes { get; } = Enum.GetValues<VectorType>().ToList();
    [ObservableProperty] private ObservableCollection<ClusterVisualItem> clusterItems = new();
    [ObservableProperty] private ObservableCollection<ImageVisualItem> imageItems = new();
    [ObservableProperty] private double canvasWidth = 1000;
    [ObservableProperty] private double canvasHeight = 1000;
    [ObservableProperty][NotifyCanExecuteChangedFor(nameof(RecalculatePcaCommand))] private bool isPcaComputing;
    [ObservableProperty] private int pcaProgress;
    [ObservableProperty][NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))] private bool isClusterComputing;
    [ObservableProperty] private float similarityThreshold = 0.85f;
    [ObservableProperty] private int sparseTopN = 2048;
    [ObservableProperty] private bool useGpu = true;
    [ObservableProperty] private int threadCount = 0;
    [ObservableProperty] private bool gpuAvailable;
    [ObservableProperty] private string gpuName = "Detecting...";
    [ObservableProperty] private int vectorCount;
    [ObservableProperty] private int clusterCount;
    [ObservableProperty] private string databaseSizeText = "0 KB";
    [ObservableProperty] private string lastScanDuration = "-";
    [ObservableProperty] private ObservableCollection<string> consoleLines = new();
    [ObservableProperty] private bool isConsoleExpanded = true;
    [ObservableProperty] private bool isCockpitExpanded = false;
    [ObservableProperty] private string themeIcon = "Light";

    private CancellationTokenSource? cts;

    public MainViewModel(
        ImageScanner imageScanner, IVectorDatabase vectorDatabase,
        ClusteringService clusteringService, StorageService storageService,
        ThemeService themeService, LogService logService)
    {
        this.imageScanner = imageScanner;
        this.vectorDatabase = vectorDatabase;
        this.clusteringService = clusteringService;
        this.storageService = storageService;
        this.themeService = themeService;
        this.logService = logService;

        logService.LogAdded += line =>
            Application.Current?.Dispatcher.InvokeAsync(() =>
            {
                ConsoleLines.Add(line);
                while (ConsoleLines.Count > LogService.MaxLines)
                    ConsoleLines.RemoveAt(0);
            });

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

    [RelayCommand] private void ToggleTheme() { themeService.ToggleTheme(); UpdateThemeIcon(); logService.Log($"Theme: {themeService.CurrentTheme}"); }
    private void UpdateThemeIcon() => ThemeIcon = themeService.CurrentTheme == ThemeService.Theme.Dark ? "Dark" : "Light";
    [RelayCommand] private void ToggleConsole() => IsConsoleExpanded = !IsConsoleExpanded;
    [RelayCommand] private void ClearConsole() { ConsoleLines.Clear(); logService.Clear(); }
    [RelayCommand] private void ToggleCockpit() { IsCockpitExpanded = !IsCockpitExpanded; logService.Log($"Cockpit: {(IsCockpitExpanded ? "expanded" : "collapsed")}"); }

    partial void OnSparseTopNChanged(int value) { logService.Log($"Compression: Top-{value}/2048"); SaveSettings(); }
    partial void OnSimilarityThresholdChanged(float value) { logService.Log($"Threshold: {value:F2}"); SaveSettings(); }
    partial void OnUseGpuChanged(bool value) { logService.Log($"Use GPU: {value}"); SaveSettings(); }
    partial void OnIsConsoleExpandedChanged(bool value) => SaveSettings();

    private void SaveSettings()
    {
        var s = AppSettings.Load();
        s.SparseTopN = SparseTopN; s.UseGpu = UseGpu; s.ThreadCount = ThreadCount;
        s.SimilarityThreshold = SimilarityThreshold; s.IsConsoleExpanded = IsConsoleExpanded;
        AppSettings.Save(s);
    }

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
            logService.Log($"Scanning: {folder}");
            await foreach (var prog in imageScanner.ScanFolderAsync(folder, SelectedVectorType, cts.Token))
            {
                CurrentFile = Path.GetFileName(prog.CurrentFile);
                ProcessedCount = prog.ProcessedCount;
                TotalCount = prog.TotalCount;
                Progress = (double)ProcessedCount / TotalCount * 100;
                if (ProcessedCount % 50 == 0)
                    logService.Log($"Vectorizing: {ProcessedCount}/{TotalCount}");
            }
            sw.Stop();
            LastScanDuration = $"{sw.Elapsed.TotalSeconds:F1}s";
            logService.Log($"Scan complete - {ProcessedCount} vectors in {LastScanDuration}");
            await LoadAndDisplayAsync();
        }
        finally { IsScanning = false; cts?.Cancel(); cts?.Dispose(); UpdateDatabaseSize(); }
    }

    [RelayCommand] private async Task LoadExistingClustersAsync() { logService.Log("Loading DB..."); await LoadAndDisplayAsync(); }
    [RelayCommand] private void CancelScan() { cts?.Cancel(); logService.Log("Scan cancelled."); }

    [RelayCommand]
    private async Task ClearAllDataAsync()
    {
        var r = MessageBox.Show(
            "This will permanently delete all stored vectors, thumbnails, and cached positions.\n\nYour original image files will NOT be affected.\n\nContinue?",
            "Clear all data", MessageBoxButton.YesNo, MessageBoxImage.Warning);
        if (r != MessageBoxResult.Yes) return;
        logService.Log("Closing DB...");
        await vectorDatabase.CloseAsync();
        await storageService.ClearAllDataAsync();
        await vectorDatabase.ReopenAsync(storageService.DatabasePath);
        Clusters.Clear(); ClusterItems.Clear(); ImageItems.Clear();
        CurrentFile = ""; ProcessedCount = 0; TotalCount = 0; Progress = 0; VectorCount = 0; ClusterCount = 0;
        UpdateDatabaseSize();
        logService.Log("All data cleared. DB reconnected.");
    }

    [RelayCommand(CanExecute = nameof(CanRecalculatePca))]
    private async Task RecalculatePcaAsync()
    {
        logService.Log("Clearing PCA cache...");
        await vectorDatabase.ClearPcaCacheAsync();
        var vectors = await vectorDatabase.GetAllAsync();
        if (vectors.Count == 0) { logService.Log("No vectors."); return; }
        await ComputeAndCachePcaAsync(vectors);
    }
    private bool CanRecalculatePca() => !IsPcaComputing && !IsScanning;

    [RelayCommand(CanExecute = nameof(CanComputeClusters))]
    private async Task ComputeClustersAsync()
    {
        IsClusterComputing = true;
        logService.Log($"Clustering (threshold {SimilarityThreshold:F2})...");
        try
        {
            var vectors = await vectorDatabase.GetAllAsync();
            var list = await Task.Run(() => clusteringService.ClusterBySimilarity(vectors, SimilarityThreshold));
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                Clusters.Clear();
                foreach (var c in list) Clusters.Add(c);
                ClusterCount = Clusters.Count;
            });
            logService.Log($"Done - {ClusterCount} clusters.");
        }
        finally { IsClusterComputing = false; }
    }
    private bool CanComputeClusters() => !IsClusterComputing && !IsScanning && !IsPcaComputing;

    // =========================================================================
    // CORE DISPLAY
    // =========================================================================

    private async Task LoadAndDisplayAsync()
    {
        var vectors = await vectorDatabase.GetAllAsync();
        VectorCount = vectors.Count;
        if (vectors.Count == 0) { logService.Log("Empty - scan a folder first."); return; }
        logService.Log($"Loaded {vectors.Count} vectors.");
        bool cacheOk = vectors.All(v => v.PcaX.HasValue && v.PcaY.HasValue);
        if (cacheOk) { logService.Log("PCA cache hit - fast render."); await PopulateFromCacheAsync(vectors); }
        else { logService.Log($"PCA cache incomplete - RSVD required."); await ComputeAndCachePcaAsync(vectors); }
        UpdateDatabaseSize();
    }

    /// <summary>
    /// Dot radius based on log-scale file size relative to dataset median.
    /// Median maps to radius 14; range is 6..22 (1.5x smaller/larger).
    /// </summary>
    private static double[] ComputeDotRadii(IReadOnlyList<long> fileSizes)
    {
        if (fileSizes.Count == 0) return Array.Empty<double>();
        var sorted = fileSizes.OrderBy(x => x).ToList();
        double median = sorted[sorted.Count / 2];
        if (median <= 0) median = 1;
        const double baseR = 14.0, minR = 6.0, maxR = 22.0;
        return fileSizes.Select(sz =>
        {
            double ratio = Math.Log10(Math.Max(sz, 1)) / Math.Log10(median);
            return Math.Clamp(baseR * ratio, minR, maxR);
        }).ToArray();
    }

    private async Task PopulateFromCacheAsync(List<ImageVector> vectors)
    {
        var minX = vectors.Min(v => v.PcaX!.Value);
        var maxX = vectors.Max(v => v.PcaX!.Value);
        var minY = vectors.Min(v => v.PcaY!.Value);
        var maxY = vectors.Max(v => v.PcaY!.Value);
        double rX = Math.Max(maxX - minX, 0.0001);
        double rY = Math.Max(maxY - minY, 0.0001);
        double pad = 0.05;
        double wW = CanvasWidth * (1 - 2 * pad);
        double wH = CanvasHeight * (1 - 2 * pad);

        var fileSizes = vectors.Select(v => v.FileSize).ToList();
        var radii = ComputeDotRadii(fileSizes);

        // Build items fully on background thread - no UI work until done
        var items = await Task.Run(() =>
        {
            var list = new List<ImageVisualItem>(vectors.Count);
            for (int i = 0; i < vectors.Count; i++)
            {
                var v = vectors[i];
                list.Add(new ImageVisualItem
                {
                    FilePath = v.FilePath,
                    ThumbnailPath = v.ThumbnailPath ?? v.FilePath,
                    FileSize = v.FileSize,
                    DotRadius = radii[i],
                    X = (v.PcaX!.Value - minX) / rX * wW + CanvasWidth * pad,
                    Y = (v.PcaY!.Value - minY) / rY * wH + CanvasHeight * pad
                });
            }
            return list;
        });

        await Application.Current.Dispatcher.InvokeAsync(() =>
        {
            ImageItems.Clear(); ClusterItems.Clear();
            foreach (var item in items) ImageItems.Add(item);
        });
        logService.Log($"Map rendered - {ImageItems.Count} images.");
    }

    private async Task ComputeAndCachePcaAsync(List<ImageVector> vectors)
    {
        IsPcaComputing = true;
        PcaProgress = 0;
        int dim = vectors.Count > 0 ? vectors[0].Vector.Length : 0;
        logService.Log($"RSVD on {vectors.Count} x {dim} (Top-{SparseTopN})...");

        var prog = new Progress<(int current, int total, string message)>(p =>
        {
            PcaProgress = p.total > 0 ? p.current * 100 / p.total : 0;
            if (!string.IsNullOrEmpty(p.message)) logService.Log(p.message);
        });

        List<ClusterPosition> positions;
        try
        {
            // Fully background - UI stays interactive during all RSVD steps
            positions = await Task.Run(() => clusteringService.CalculatePositionsSparse(
                new List<ImageCluster> { new ImageCluster { Images = vectors, ClusterId = 0 } },
                (int)CanvasWidth, (int)CanvasHeight, SparseTopN, prog));
        }
        finally { IsPcaComputing = false; }

        var nonCentroids = positions.Where(p => !p.IsCentroid && p.ImageVector != null).ToList();
        var radii = ComputeDotRadii(nonCentroids.Select(p => p.ImageVector.FileSize).ToList());

        logService.Log($"RSVD done - rendering {nonCentroids.Count} dots...");

        // Build visual items off-thread
        var items = await Task.Run(() =>
        {
            var list = new List<ImageVisualItem>(nonCentroids.Count);
            for (int i = 0; i < nonCentroids.Count; i++)
            {
                var pos = nonCentroids[i];
                list.Add(new ImageVisualItem
                {
                    ClusterId = pos.ClusterId,
                    FilePath = pos.ImageVector.FilePath,
                    ThumbnailPath = pos.ImageVector.ThumbnailPath ?? pos.ImageVector.FilePath,
                    FileSize = pos.ImageVector.FileSize,
                    DotRadius = radii.Length > i ? radii[i] : 14.0,
                    X = pos.X,
                    Y = pos.Y
                });
            }
            return list;
        });

        // Single UI update
        await Application.Current.Dispatcher.InvokeAsync(() =>
        {
            ImageItems.Clear(); ClusterItems.Clear();
            foreach (var item in items) ImageItems.Add(item);
        });
        logService.Log($"Map rendered - {ImageItems.Count} images.");

        // Save PCA coordinates in background without blocking UI
        _ = Task.Run(async () =>
        {
            var tasks = nonCentroids
                .Select(p => vectorDatabase.SavePcaCoordinatesAsync(p.ImageVector.FilePath, (float)p.X, (float)p.Y))
                .ToList();
            await Task.WhenAll(tasks);
            logService.Log($"Cached PCA for {tasks.Count} images.");
        });
    }

    private void UpdateDatabaseSize()
    {
        try
        {
            var dbPath = storageService.DatabasePath;
            DatabaseSizeText = File.Exists(dbPath)
                ? new FileInfo(dbPath).Length is long b
                    ? b < 1_048_576 ? $"{b / 1024.0:F1} KB" : $"{b / 1_048_576.0:F1} MB"
                    : "0 KB"
                : "0 KB";
        }
        catch { DatabaseSizeText = "N/A"; }
    }
}
