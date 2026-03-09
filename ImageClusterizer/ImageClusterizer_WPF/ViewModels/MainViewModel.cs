namespace ImageClusterizer.ViewModels;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Models;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
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

    [ObservableProperty][NotifyPropertyChangedFor(nameof(ProcessedCountText))]
    private int processedCount;
    public string ProcessedCountText => ProcessedCount.ToString();

    [ObservableProperty][NotifyPropertyChangedFor(nameof(TotalCountText))]
    private int totalCount;
    public string TotalCountText => TotalCount.ToString();

    [ObservableProperty] private double progress;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsNotScanning), nameof(ProgressVisibility), nameof(IsIdle), nameof(IsProcessing))]
    [NotifyCanExecuteChangedFor(nameof(RecalculatePcaCommand))]
    [NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))]
    private bool isScanning;

    public bool IsNotScanning => !IsScanning;
    public Visibility ProgressVisibility => IsScanning ? Visibility.Visible : Visibility.Collapsed;

    [ObservableProperty] private VectorType selectedVectorType = VectorType.Embedding;
    public IReadOnlyList<VectorType> AvailableVectorTypes { get; } = Enum.GetValues<VectorType>().ToList();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasDots))]
    private IReadOnlyList<MapDot> mapDots = Array.Empty<MapDot>();

    public bool HasDots => MapDots.Count > 0;

    [ObservableProperty] private double canvasWidth = 1000;
    [ObservableProperty] private double canvasHeight = 1000;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RecalculatePcaCommand))]
    [NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))]
    [NotifyPropertyChangedFor(nameof(IsIdle), nameof(IsProcessing))]
    private bool isPcaComputing;

    [ObservableProperty] private int pcaProgress;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))]
    [NotifyPropertyChangedFor(nameof(IsIdle), nameof(IsProcessing), nameof(ClusteringStatus))]
    private bool isClusterComputing;

    public bool IsIdle => !IsScanning && !IsPcaComputing && !IsClusterComputing;
    public bool IsProcessing => IsScanning || IsPcaComputing || IsClusterComputing;
    public string ClusteringStatus => IsClusterComputing ? "Computing clusters..." : ClusterCount > 0 ? $"{ClusterCount} clusters" : "No clusters";

    [ObservableProperty] private float similarityThreshold = 0.85f;
    [ObservableProperty] private int sparseTopN = 2048;
    [ObservableProperty] private bool useGpu = true;
    [ObservableProperty] private int threadCount = 0;
    [ObservableProperty] private bool gpuAvailable;
    [ObservableProperty] private string gpuName = "Detecting...";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RecalculatePcaCommand))]
    [NotifyCanExecuteChangedFor(nameof(ComputeClustersCommand))]
    [NotifyPropertyChangedFor(nameof(HasVectors))]
    private int vectorCount;

    public bool HasVectors => VectorCount > 0;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ClusteringStatus))]
    private int clusterCount;

    [ObservableProperty] private string databaseSizeText = "0 KB";
    [ObservableProperty] private string lastScanDuration = "-";

    [ObservableProperty] private string zoomText = "100%";
    [ObservableProperty] private string currentPhase = "Idle";
    [ObservableProperty] private string elapsedText = "-";
    [ObservableProperty] private string etaText = "-";
    [ObservableProperty] private ObservableCollection<string> consoleLines = new();
    [ObservableProperty] private bool isConsoleExpanded = true;
    [ObservableProperty] private bool isCockpitExpanded = false;
    [ObservableProperty] private string themeIcon = "Light";

    [ObservableProperty] private bool isScanPhaseActive = false;
    [ObservableProperty] private bool isPcaPhaseActive = false;
    [ObservableProperty] private bool isRenderPhaseActive = false;
    [ObservableProperty] private bool isClusterPhaseActive = false;
    [ObservableProperty] private string scanPhaseColor = "#555555";
    [ObservableProperty] private string pcaPhaseColor = "#555555";
    [ObservableProperty] private string renderPhaseColor = "#555555";
    [ObservableProperty] private string clusterPhaseColor = "#555555";
    [ObservableProperty] private int itemsPerSecond = 0;

    private int _lastProgressCount = 0;
    private DateTime _lastProgressTime = DateTime.Now;

    [ObservableProperty] private ObservableCollection<AnalysisProfile> profiles = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanDeleteProfile))]
    private AnalysisProfile? selectedProfile;

    [ObservableProperty] private string newProfileName = "";
    public bool CanDeleteProfile => SelectedProfile != null && SelectedProfile.Name != "Default";

    private readonly System.Timers.Timer _elapsedTimer = new(1000);
    private Stopwatch? _phaseStopwatch;
    private int _etaTotal;
    private int _etaCurrent;
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

        logService.LogAdded += line => Application.Current?.Dispatcher.InvokeAsync(() =>
        {
            ConsoleLines.Add(line);
            while (ConsoleLines.Count > LogService.MaxLines) ConsoleLines.RemoveAt(0);
        });

        _elapsedTimer.Elapsed += OnElapsedTick;
        _elapsedTimer.AutoReset = true;

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

        foreach (var p in settings.Profiles)
            Profiles.Add(p);
        SelectedProfile = Profiles.FirstOrDefault(p => p.Name == settings.LastUsedProfile) ?? Profiles.FirstOrDefault();

        UpdateThemeIcon();
        logService.Log("ImageClusterizer started.");
        UpdateDatabaseSize();
    }

    private void OnElapsedTick(object? sender, System.Timers.ElapsedEventArgs e)
    {
        if (_phaseStopwatch == null) return;
        var el = _phaseStopwatch.Elapsed;
        string elapsed = el.TotalMinutes >= 1
            ? $"{(int)el.TotalMinutes}m {el.Seconds:D2}s"
            : $"{el.TotalSeconds:F1}s";

        string eta = "-";
        if (_etaTotal > 0 && _etaCurrent > 0 && _etaCurrent < _etaTotal)
        {
            double rate = _etaCurrent / el.TotalSeconds;
            if (rate > 0)
            {
                double remaining = (_etaTotal - _etaCurrent) / rate;
                eta = remaining >= 60
                    ? $"~{(int)(remaining / 60)}m {(int)(remaining % 60):D2}s"
                    : $"~{remaining:F0}s";
            }
        }

        int ips = 0;
        var now = DateTime.Now;
        double secSinceLast = (now - _lastProgressTime).TotalSeconds;
        if (secSinceLast >= 1.0 && _etaCurrent > _lastProgressCount)
        {
            ips = (int)((_etaCurrent - _lastProgressCount) / secSinceLast);
            _lastProgressCount = _etaCurrent;
            _lastProgressTime = now;
        }

        Application.Current?.Dispatcher.InvokeAsync(() =>
        {
            ElapsedText = elapsed;
            EtaText = eta;
            if (ips > 0) ItemsPerSecond = ips;
        });
    }

    private void StartPhase(string name, int total = 0)
    {
        _etaTotal = total;
        _etaCurrent = 0;
        _lastProgressCount = 0;
        _lastProgressTime = DateTime.Now;
        _phaseStopwatch = Stopwatch.StartNew();
        _elapsedTimer.Start();

        bool scan = name.StartsWith("Scan", StringComparison.OrdinalIgnoreCase);
        bool pca = name.Contains("PCA", StringComparison.OrdinalIgnoreCase) || name.Contains("RSVD", StringComparison.OrdinalIgnoreCase);
        bool render = name.StartsWith("Render", StringComparison.OrdinalIgnoreCase);
        bool cluster = name.StartsWith("Cluster", StringComparison.OrdinalIgnoreCase);

        Application.Current?.Dispatcher.InvokeAsync(() =>
        {
            CurrentPhase = name;
            ElapsedText = "0s";
            EtaText = "-";
            ItemsPerSecond = 0;
            IsScanPhaseActive = scan;
            IsPcaPhaseActive = pca;
            IsRenderPhaseActive = render;
            IsClusterPhaseActive = cluster;
            ScanPhaseColor = scan ? "#00C853" : "#555555";
            PcaPhaseColor = pca ? "#FF9100" : "#555555";
            RenderPhaseColor = render ? "#40C4FF" : "#555555";
            ClusterPhaseColor = cluster ? "#E040FB" : "#555555";
        });
    }

    private void UpdatePhaseProgress(int current) { _etaCurrent = current; }

    private void StopPhase()
    {
        _elapsedTimer.Stop();
        var el = _phaseStopwatch?.Elapsed ?? TimeSpan.Zero;
        _phaseStopwatch = null;
        string elapsed = el.TotalMinutes >= 1
            ? $"{(int)el.TotalMinutes}m {el.Seconds:D2}s"
            : $"{el.TotalSeconds:F1}s";

        Application.Current?.Dispatcher.InvokeAsync(() =>
        {
            CurrentPhase = "Idle";
            ElapsedText = elapsed;
            EtaText = "Done";
            IsScanPhaseActive = false;
            IsPcaPhaseActive = false;
            IsRenderPhaseActive = false;
            IsClusterPhaseActive = false;
            ScanPhaseColor = "#555555";
            PcaPhaseColor = "#555555";
            RenderPhaseColor = "#555555";
            ClusterPhaseColor = "#555555";
            ItemsPerSecond = 0;
        });
    }

    [RelayCommand]
    private void ToggleTheme()
    {
        themeService.ToggleTheme();
        UpdateThemeIcon();
        logService.Log($"Theme: {themeService.CurrentTheme}");
    }

    private void UpdateThemeIcon() =>
        ThemeIcon = themeService.CurrentTheme == ThemeService.Theme.Dark ? "Dark" : "Light";

    [RelayCommand] private void ToggleConsole() => IsConsoleExpanded = !IsConsoleExpanded;
    [RelayCommand] private void ClearConsole() { ConsoleLines.Clear(); logService.Clear(); }
    [RelayCommand] private void ToggleCockpit()
    {
        IsCockpitExpanded = !IsCockpitExpanded;
        logService.Log($"Cockpit: {(IsCockpitExpanded ? "expanded" : "collapsed")}");
    }

    [RelayCommand]
    private void CreateProfile()
    {
        string name = NewProfileName.Trim();
        if (string.IsNullOrWhiteSpace(name)) return;
        if (Profiles.Any(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase)))
        {
            logService.Log($"Profile already exists: '{name}'");
            return;
        }
        var profile = new AnalysisProfile
        {
            Name = name,
            SparseTopN = SparseTopN,
            SimilarityThreshold = SimilarityThreshold,
            VectorType = SelectedVectorType.ToString(),
            UseGpu = UseGpu
        };
        Profiles.Add(profile);
        SelectedProfile = profile;
        NewProfileName = "";
        SaveProfilesToDisk();
        logService.Log($"Profile created: '{name}'");
    }

    [RelayCommand]
    private void SaveCurrentProfile()
    {
        if (SelectedProfile == null) return;
        SelectedProfile.SparseTopN = SparseTopN;
        SelectedProfile.SimilarityThreshold = SimilarityThreshold;
        SelectedProfile.VectorType = SelectedVectorType.ToString();
        SelectedProfile.UseGpu = UseGpu;
        SaveProfilesToDisk();
        logService.Log($"Profile saved: '{SelectedProfile.Name}'");
    }

    [RelayCommand(CanExecute = nameof(CanDeleteProfile))]
    private void DeleteProfile()
    {
        if (SelectedProfile == null || !CanDeleteProfile) return;
        string name = SelectedProfile.Name;
        Profiles.Remove(SelectedProfile);
        SelectedProfile = Profiles.FirstOrDefault();
        SaveProfilesToDisk();
        logService.Log($"Profile deleted: '{name}'");
    }

    partial void OnSelectedProfileChanged(AnalysisProfile? value)
    {
        if (value == null) return;
        SparseTopN = value.SparseTopN;
        SimilarityThreshold = (float)value.SimilarityThreshold;
        if (Enum.TryParse<VectorType>(value.VectorType, out var vt)) SelectedVectorType = vt;
        UseGpu = value.UseGpu;
        OnPropertyChanged(nameof(CanDeleteProfile));
        DeleteProfileCommand.NotifyCanExecuteChanged();
        var s = AppSettings.Load();
        s.LastUsedProfile = value.Name;
        AppSettings.Save(s);
        logService.Log($"Profile applied: '{value.Name}'");
    }

    private void SaveProfilesToDisk()
    {
        var s = AppSettings.Load();
        s.Profiles = Profiles.ToList();
        s.LastUsedProfile = SelectedProfile?.Name ?? "Default";
        AppSettings.Save(s);
    }

    partial void OnSparseTopNChanged(int value) { logService.Log($"Compression: Top-{value}/2048"); SaveSettings(); }
    partial void OnSimilarityThresholdChanged(float value) { logService.Log($"Threshold: {value:F2}"); SaveSettings(); }
    partial void OnUseGpuChanged(bool value) { logService.Log($"Use GPU: {value}"); SaveSettings(); }
    partial void OnIsConsoleExpandedChanged(bool value) => SaveSettings();

    private void SaveSettings()
    {
        var s = AppSettings.Load();
        s.SparseTopN = SparseTopN;
        s.UseGpu = UseGpu;
        s.ThreadCount = ThreadCount;
        s.SimilarityThreshold = SimilarityThreshold;
        s.IsConsoleExpanded = IsConsoleExpanded;
        AppSettings.Save(s);
    }

    [RelayCommand]
    private async Task StartScanImagesAsync()
    {
        string? folder = Utility.SelectFolderDiagoAsync();
        if (string.IsNullOrWhiteSpace(folder)) return;

        IsScanning = true;
        cts = new CancellationTokenSource();
        StartPhase("Scanning", 0);
        var sw = Stopwatch.StartNew();

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
                UpdatePhaseProgress(prog.ProcessedCount);
                if (ProcessedCount % 50 == 0)
                    logService.Log($"Vectorizing: {ProcessedCount}/{TotalCount}");
            }

            sw.Stop();
            LastScanDuration = $"{sw.Elapsed.TotalSeconds:F1}s";
            logService.Log($"Scan complete - {ProcessedCount} vectors in {LastScanDuration}");
            await LoadAndDisplayAsync();
        }
        catch (OperationCanceledException)
        {
            logService.Log("Scan cancelled by user.");
        }
        catch (Exception ex)
        {
            logService.Log($"Scan error: {ex.Message}");
        }
        finally
        {
            IsScanning = false;
            StopPhase();
            cts?.Cancel();
            cts?.Dispose();
            UpdateDatabaseSize();
            RecalculatePcaCommand.NotifyCanExecuteChanged();
            ComputeClustersCommand.NotifyCanExecuteChanged();
        }
    }

    [RelayCommand] private async Task LoadExistingClustersAsync()
    {
        logService.Log("Loading DB...");
        await LoadAndDisplayAsync();
    }

    [RelayCommand] private void CancelScan()
    {
        cts?.Cancel();
        logService.Log("Scan cancelled.");
    }

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

        Clusters.Clear();
        MapDots = Array.Empty<MapDot>();
        CurrentFile = "";
        ProcessedCount = 0;
        TotalCount = 0;
        Progress = 0;
        VectorCount = 0;
        ClusterCount = 0;
        UpdateDatabaseSize();
        RecalculatePcaCommand.NotifyCanExecuteChanged();
        ComputeClustersCommand.NotifyCanExecuteChanged();
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

    private bool CanRecalculatePca() =>
        !IsPcaComputing && !IsScanning && VectorCount > 0;

    [RelayCommand(CanExecute = nameof(CanComputeClusters))]
    private async Task ComputeClustersAsync()
    {
        IsClusterComputing = true;
        StartPhase("Clustering");
        logService.Log($"Clustering (threshold {SimilarityThreshold:F2})...");

        try
        {
            var vectors = await vectorDatabase.GetAllAsync();
            var validVectors = vectors
                .Where(v => v.Vector != null && v.Vector.Length > 0)
                .ToList();

            if (validVectors.Count == 0)
            {
                logService.Log("No valid vectors for clustering.");
                return;
            }

            if (validVectors.Count < vectors.Count)
                logService.Log($"Warning: {vectors.Count - validVectors.Count} vectors skipped (null/empty).");

            var list = await Task.Run(() =>
                clusteringService.ClusterBySimilarity(validVectors, SimilarityThreshold));

            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                Clusters.Clear();
                foreach (var c in list) Clusters.Add(c);
                ClusterCount = Clusters.Count;
            });

            logService.Log($"Done - {ClusterCount} clusters from {validVectors.Count} images.");
        }
        catch (Exception ex)
        {
            logService.Log($"Clustering error: {ex.GetType().Name}: {ex.Message}");
        }
        finally
        {
            IsClusterComputing = false;
            StopPhase();
        }
    }

    private bool CanComputeClusters() =>
        !IsClusterComputing && !IsScanning && !IsPcaComputing && VectorCount > 0;

    private async Task LoadAndDisplayAsync()
    {
        var vectors = await vectorDatabase.GetAllAsync();
        VectorCount = vectors.Count;

        RecalculatePcaCommand.NotifyCanExecuteChanged();
        ComputeClustersCommand.NotifyCanExecuteChanged();

        if (vectors.Count == 0)
        {
            logService.Log("Empty - scan a folder first.");
            return;
        }

        logService.Log($"Loaded {vectors.Count} vectors.");
        bool cacheOk = vectors.All(v => v.PcaX.HasValue && v.PcaY.HasValue);

        if (cacheOk)
        {
            logService.Log("PCA cache hit - fast render.");
            await PopulateFromCacheAsync(vectors);
        }
        else
        {
            logService.Log($"PCA cache incomplete ({vectors.Count(v => !v.PcaX.HasValue)} missing) - RSVD required.");
            await ComputeAndCachePcaAsync(vectors);
        }

        UpdateDatabaseSize();
    }

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
        StartPhase("Rendering (cache)", vectors.Count);
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

        var dots = await Task.Run(() =>
        {
            var list = new List<MapDot>(vectors.Count);
            for (int i = 0; i < vectors.Count; i++)
            {
                var v = vectors[i];
                list.Add(new MapDot
                {
                    FilePath = v.FilePath,
                    ThumbnailPath = v.ThumbnailPath ?? v.FilePath,
                    FileSize = v.FileSize,
                    Radius = radii[i],
                    X = (v.PcaX!.Value - minX) / rX * wW + CanvasWidth * pad,
                    Y = (v.PcaY!.Value - minY) / rY * wH + CanvasHeight * pad
                });
            }
            return (IReadOnlyList<MapDot>)list.AsReadOnly();
        });

        MapDots = dots;
        StopPhase();
        logService.Log($"Map rendered - {dots.Count} images from cache.");
    }

    private async Task ComputeAndCachePcaAsync(List<ImageVector> vectors)
    {
        IsPcaComputing = true;
        PcaProgress = 0;
        int dim = vectors.Count > 0 ? vectors[0].Vector.Length : 0;
        logService.Log($"RSVD on {vectors.Count} x {dim} (Top-{SparseTopN})...");
        StartPhase("RSVD / PCA", vectors.Count);

        var prog = new Progress<(int current, int total, string message)>(p =>
        {
            PcaProgress = p.total > 0 ? p.current * 100 / p.total : 0;
            UpdatePhaseProgress(p.current);
            if (!string.IsNullOrEmpty(p.message)) logService.Log(p.message);
        });

        List<ClusterPosition> positions;
        try
        {
            positions = await Task.Run(() =>
                clusteringService.CalculatePositionsSparse(
                    new List<ImageCluster> { new ImageCluster { Images = vectors, ClusterId = 0 } },
                    (int)CanvasWidth, (int)CanvasHeight, SparseTopN, prog));
        }
        catch (Exception ex)
        {
            logService.Log($"PCA error: {ex.GetType().Name}: {ex.Message}");
            IsPcaComputing = false;
            StopPhase();
            return;
        }
        finally
        {
            IsPcaComputing = false;
        }

        var nonCentroids = positions.Where(p => !p.IsCentroid && p.ImageVector != null).ToList();
        var radii = ComputeDotRadii(nonCentroids.Select(p => p.ImageVector.FileSize).ToList());
        logService.Log($"RSVD done - building {nonCentroids.Count} dots...");
        StartPhase("Rendering", nonCentroids.Count);

        var dots = await Task.Run(() =>
        {
            var list = new List<MapDot>(nonCentroids.Count);
            for (int i = 0; i < nonCentroids.Count; i++)
            {
                var pos = nonCentroids[i];
                list.Add(new MapDot
                {
                    FilePath = pos.ImageVector.FilePath,
                    ThumbnailPath = pos.ImageVector.ThumbnailPath ?? pos.ImageVector.FilePath,
                    FileSize = pos.ImageVector.FileSize,
                    Radius = radii.Length > i ? radii[i] : 14.0,
                    X = pos.X,
                    Y = pos.Y
                });
            }
            return (IReadOnlyList<MapDot>)list.AsReadOnly();
        });

        MapDots = dots;
        StopPhase();
        logService.Log($"Map rendered - {dots.Count} images.");

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
