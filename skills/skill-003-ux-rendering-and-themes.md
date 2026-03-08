# SKILL-003: UX Rendering, Themes, Console Log, Sparse Compression and Cockpit Controls

## Goal

Transform the application into a fully responsive, information-rich tool that:
- Never freezes the UI during any computation
- Shows a live monospaced console log panel so the user always knows what is happening
- Displays progressive lazy rendering: images appear on the map as PCA results arrive
- Supports dark/light theme toggle with persistent preference
- Exposes a vector compression quality slider (Sparse Top-N) for speed vs accuracy trade-off
- Auto-detects GPU availability for ONNX inference and shows it as a live indicator
- Has a "cockpit" style advanced settings panel with labeled indicators, toggles, and live telemetry

## Depends On

- SKILL-001 must be completed (core pipeline)
- SKILL-002 must be completed (thumbnails + PCA cache + StorageService)

## Scope

- Work only inside: `ImageClusterizer/ImageClusterizer_WPF/`
- Do NOT touch: `Polygon/` folder
- Do NOT change Channel-based batch processing in `ImageScanner.cs`
- Do NOT change clustering algorithm internals in `ClusteringService.cs`

## Stack

- .NET 8, WPF (net8.0-windows10.0.17763.0)
- CommunityToolkit.Mvvm 8.4.0
- WPF ResourceDictionary for themes (no third-party theme libraries)
- System.Text.Json (already in .NET 8 BCL, no extra NuGet)
- Microsoft.ML.OnnxRuntime already in project (for GPU detection)

---

## What to implement

### 1. Dark / Light Theme

**Files to create:**
- `Themes/LightTheme.xaml`
- `Themes/DarkTheme.xaml`

Each ResourceDictionary must define these named brushes:
```
AppBackground        — main window background
AppSurface           — panel/card background (slightly different from background)
AppBorder            — border/separator color
AppText              — primary text
AppTextSecondary     — dimmed/secondary text
AppAccent            — accent color for buttons, highlights, ellipses
AppAccentHover       — accent on hover
ToolBarBackground    — toolbar strip background
StatusBarBackground  — status bar background
ConsoleBackground    — console log panel background (near-black for both themes)
ConsoleForeground    — console text color (green-ish for dark, green for light)
IndicatorActive      — LED-style indicator active color (bright green)
IndicatorInactive    — LED-style indicator inactive color (dark gray)
IndicatorWarning     — LED-style indicator warning color (amber)
```

**Dark theme palette suggestion:**
- AppBackground: #1E1E1E, AppSurface: #252526, AppBorder: #3F3F46
- AppText: #D4D4D4, AppTextSecondary: #808080, AppAccent: #007ACC, AppAccentHover: #1C97EA
- ToolBarBackground: #2D2D30, StatusBarBackground: #007ACC
- ConsoleBackground: #0C0C0C, ConsoleForeground: #00FF41 (matrix green)
- IndicatorActive: #00C853, IndicatorInactive: #424242, IndicatorWarning: #FFB300

**Light theme palette suggestion:**
- AppBackground: #F3F3F3, AppSurface: #FFFFFF, AppBorder: #CCCCCC
- AppText: #1E1E1E, AppTextSecondary: #717171, AppAccent: #0078D4, AppAccentHover: #106EBE
- ToolBarBackground: #F0F0F0, StatusBarBackground: #0078D4
- ConsoleBackground: #1E1E1E, ConsoleForeground: #4EC94E
- IndicatorActive: #00C853, IndicatorInactive: #BDBDBD, IndicatorWarning: #FF8F00

**ThemeService** (new singleton at `Services/ThemeService.cs`):
```csharp
public class ThemeService
{
    public enum Theme { Light, Dark }
    public Theme CurrentTheme { get; private set; } = Theme.Light;

    public void ToggleTheme() { ... }
    public void ApplyTheme(Theme theme) { ... }
    public void SavePreference() { ... }
    public void LoadPreference() { ... }
}
```

AppSettings.json format (next to executable):
```json
{
  "Theme": "Dark",
  "SparseTopN": 2048,
  "UseGpu": true,
  "ThreadCount": 0,
  "SimilarityThreshold": 0.85,
  "IsConsoleExpanded": true
}
```

Register ThemeService as singleton in App.xaml.cs. Call LoadPreference() on app startup before MainWindow is shown. Apply theme brushes using DynamicResource throughout MainWindow.xaml.

---

### 2. Console Log Panel

Add a collapsible console log panel at the bottom of the main window (above the status bar).

**Visual requirements:**
- Monospaced font: Consolas, font size 11
- Background: {DynamicResource ConsoleBackground}
- Foreground: {DynamicResource ConsoleForeground}
- Fixed height 120px scrollable (ScrollViewer with auto-scroll to bottom)
- Header row: "Console" label + "Clear" button + collapse toggle
- Collapsible: collapsed = 24px header only, expanded = 120px
- Max 200 log lines (oldest removed when limit exceeded)

**LogService** (new singleton at `Services/LogService.cs`):
```csharp
public class LogService
{
    public event Action<string>? LogAdded;
    public void Log(string message) { ... }  // adds "[HH:mm:ss.fff] message"
    public void Clear() { ... }
}
```

Add to MainViewModel:
```csharp
[ObservableProperty] private ObservableCollection<string> consoleLines = new();
[ObservableProperty] private bool isConsoleExpanded = true;
```

Subscribe to LogService.LogAdded in constructor, dispatch to UI thread, trim to 200 lines.

Call LogService.Log() at every important step:
- Folder scan start/complete
- GPU detection result
- Vectorization progress (every N files)
- PCA start/complete with matrix dimensions
- Progressive batch rendering updates
- Clustering start/complete
- DB size updates
- Settings changes

---

### 3. Vector Compression (Sparse Top-N Slider)

Based on Polygon/5. ResNet50_Sparse_Dot_Product_test: full 2048D vectors can be compressed
to top-N values (sparse representation) before PCA. This drastically reduces SVD computation
time with minor accuracy loss at aggressive settings.

**New methods to ADD to ClusteringService** (do NOT modify existing methods):

```csharp
/// <summary>
/// Converts a dense embedding to sparse representation by keeping only top-N values.
/// Based on Polygon/5 experiments. sparseTopN=2048 means no compression (full vector).
/// sparseTopN=10 means extreme compression (10 out of 2048 values kept).
/// </summary>
public static float[] ToSparse(float[] vector, int sparseTopN)
{
    if (sparseTopN <= 0 || sparseTopN >= vector.Length)
        return vector;
    var result = new float[vector.Length];
    var topIndices = vector
        .Select((v, i) => (index: i, absValue: MathF.Abs(v)))
        .OrderByDescending(x => x.absValue)
        .Take(sparseTopN)
        .Select(x => x.index);
    foreach (var idx in topIndices)
        result[idx] = vector[idx];
    return result;
}

/// <summary>
/// CalculatePositions variant that applies sparse compression before PCA.
/// sparseTopN=2048 is equivalent to no compression.
/// Reports (current, total, message) progress tuples via IProgress.
/// </summary>
public List<ClusterPosition> CalculatePositionsSparse(
    List<ImageCluster> clusters,
    int canvasWidth,
    int canvasHeight,
    int sparseTopN,
    IProgress<(int current, int total, string message)>? progress = null)
```

**UI Slider** (in Advanced Settings panel):
- Range: 8 to 2048, default 2048
- Label: "Vector compression: Top-{N} / 2048"
- Quality description updates dynamically:
  - 2048: "Full quality (slowest)"
  - 512-1024: "High quality"
  - 128-256: "Balanced"
  - 32-64: "Fast (reduced accuracy)"
  - 8-16: "Extreme compression (fastest)"

Add to MainViewModel:
```csharp
[ObservableProperty] private int sparseTopN = 2048;
```

---

### 4. GPU Detection and LED Indicator

At app startup detect GPU via OrtEnv:

```csharp
// Utility/GpuDetector.cs
public static class GpuDetector
{
    public record GpuInfo(bool IsAvailable, string ProviderName, string DeviceName);

    public static GpuInfo Detect()
    {
        try {
            var providers = OrtEnv.Instance().GetAvailableProviders();
            if (providers.Contains("CUDAExecutionProvider"))
                return new GpuInfo(true, "CUDA", "CUDA GPU");
            if (providers.Contains("DmlExecutionProvider"))
                return new GpuInfo(true, "DirectML", "GPU (DirectML)");
        } catch { }
        return new GpuInfo(false, "CPU", "No GPU detected");
    }
}
```

Add to MainViewModel:
```csharp
[ObservableProperty] private bool gpuAvailable;
[ObservableProperty] private string gpuName = "Detecting...";
[ObservableProperty] private bool useGpu = true;
```

LED indicator in cockpit panel (Ellipse 10x10):
- Green (IndicatorActive): GPU available + UseGpu=true
- Amber (IndicatorWarning): GPU available but UseGpu=false
- Red (IndicatorInactive): no GPU detected

Note: actual GPU session wiring in OnnxRuntime is for a future SKILL. This SKILL only shows detection.

---

### 5. Non-Blocking UI and Progressive Rendering

**PCA — fully async with IProgress:**

Modify ComputeAndCachePcaAsync in MainViewModel:
```csharp
IsPcaComputing = true;
PcaProgress = 0;
logService.Log($"Starting PCA (SVD on {vectors.Count} x {dim} matrix)...");

var prog = new Progress<(int current, int total, string message)>(p => {
    PcaProgress = p.total > 0 ? p.current * 100 / p.total : 0;
    if (!string.IsNullOrEmpty(p.message)) logService.Log(p.message);
});

var positions = await Task.Run(() =>
    clusteringService.CalculatePositionsSparse(
        new List<ImageCluster> { new() { Images = vectors, ClusterId = 0 } },
        (int)CanvasWidth, (int)CanvasHeight, SparseTopN, prog));

// Progressive batch rendering — 100 items at a time
ImageItems.Clear();
const int batchSize = 100;
var nonCentroids = positions.Where(p => !p.IsCentroid).ToList();
for (int batchStart = 0; batchStart < nonCentroids.Count; batchStart += batchSize)
{
    var chunk = nonCentroids.Skip(batchStart).Take(batchSize);
    await Application.Current.Dispatcher.InvokeAsync(() => {
        foreach (var pos in chunk)
            ImageItems.Add(new ImageVisualItem { ... });
    });
    await Task.Delay(1);  // yield to UI thread
}
IsPcaComputing = false;
```

**Clustering — lazy with explicit trigger:**
```csharp
[RelayCommand(CanExecute = nameof(CanComputeClusters))]
private async Task ComputeClustersAsync()
{
    IsClusterComputing = true;
    logService.Log($"Clustering: threshold={SimilarityThreshold:F2}...");
    var vectors = await vectorDatabase.GetAllAsync();
    var clusterList = await Task.Run(() =>
        clusteringService.ClusterBySimilarity(vectors, SimilarityThreshold));
    await Application.Current.Dispatcher.InvokeAsync(() => {
        Clusters.Clear();
        foreach (var c in clusterList) Clusters.Add(c);
    });
    ClusterCount = Clusters.Count;
    logService.Log($"Clustering complete — {ClusterCount} clusters.");
    IsClusterComputing = false;
}
private bool CanComputeClusters() => !IsClusterComputing && !IsScanning && !IsPcaComputing;
```

Add new observable properties:
```csharp
[ObservableProperty] private bool isPcaComputing;
[ObservableProperty] private int pcaProgress;
[ObservableProperty] private bool isClusterComputing;
[ObservableProperty] private float similarityThreshold = 0.85f;
[ObservableProperty] private int vectorCount;
[ObservableProperty] private int clusterCount;
[ObservableProperty] private string databaseSizeText = "0 KB";
[ObservableProperty] private string lastScanDuration = "-";
```

---

### 6. Cockpit Advanced Settings Panel

Collapsible panel (between toolbar and tab control), collapsed by default.
Header shows compact summary: "Advanced  |  GPU: RTX 3080  |  Threads: 8  |  Sparse: 2048D  |  DB: 145.3 MB"

Expanded layout — 2-column grid of groups:

**Group A: Compute Hardware**
- LED Ellipse (10x10) + GPU name label
- CheckBox "Use GPU" bound to UseGpu (disabled if no GPU detected)
- Label "Threads:" + Slider (1 to ProcessorCount, step 1) + "{ThreadCount} threads"
  (0 = auto = ProcessorCount shown in parentheses)

**Group B: Vector Compression**
- Slider SparseTopN (steps: 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
- Label "Top-{SparseTopN} / 2048"
- Quality description label (computed from SparseTopN value)

**Group C: Clustering Parameters**
- Slider SimilarityThreshold (0.50 to 0.99, step 0.01, tick every 0.1)
- Label "Similarity: {SimilarityThreshold:F2}"
- ToolTip: "Higher = stricter grouping (more clusters, fewer images per cluster)"

**Group D: Live Telemetry (read-only, updates live)**
- Images indexed: {VectorCount}
- Clusters found: {ClusterCount}
- Database size: {DatabaseSizeText}
- Vector type: {SelectedVectorType}
- Last scan: {LastScanDuration}

---

### 7. Tab Reorganization

**Tab 1: "Map"** — primary, opens by default
- PCA scatter canvas
- Inside tab: "Recalculate PCA" button (CanExecute = !IsPcaComputing && !IsScanning)
- ProgressBar Visibility=Visible when IsPcaComputing, Indeterminate style
- PcaProgress % label next to progress bar
- Canvas with ItemsControl bound to ImageItems

**Tab 2: "Clusters"**
- Empty state (TextBlock + icon) shown when Clusters.Count == 0 and !IsClusterComputing
- "Compute clusters" Button (CanExecute = CanComputeClusters)
- Indeterminate ProgressBar visible when IsClusterComputing
- ItemsControl / WrapPanel with image cluster cards when populated

---

### 8. Status Bar

DockPanel Dock=Bottom, height 24px, Background={DynamicResource StatusBarBackground}.

Left: StackPanel Horizontal
- "Images: {VectorCount}"
- "|"
- "Clusters: {ClusterCount}"
- "|"
- "{SelectedVectorType}"

Right: StackPanel Horizontal
- LED Ellipse (8x8) colored by GpuAvailable/UseGpu
- "{GpuName}"
- "|"
- "{DatabaseSizeText}"
- "|"
- Theme toggle button: Text = "Light" or "Dark", click = ToggleThemeCommand

---

### 9. Recalculate PCA

Add to IVectorDatabase interface:
```csharp
Task ClearPcaCacheAsync();
```

Implement in LiteDbVectorStore:
```csharp
public async Task ClearPcaCacheAsync()
{
    await Task.Run(() => {
        var all = _collection.FindAll().ToList();
        foreach (var v in all) { v.PcaX = null; v.PcaY = null; }
        foreach (var v in all) _collection.Update(v);
    });
}
```

Add to MainViewModel:
```csharp
[RelayCommand(CanExecute = nameof(CanRecalculatePca))]
private async Task RecalculatePcaAsync()
{
    logService.Log("Clearing PCA cache — forcing full SVD recompute...");
    await vectorDatabase.ClearPcaCacheAsync();
    var vectors = await vectorDatabase.GetAllAsync();
    await ComputeAndCachePcaAsync(vectors);
}
private bool CanRecalculatePca() => !IsPcaComputing && !IsScanning;
```

---

## Constraints

- No third-party NuGet packages for themes
- Add ToSparse() and CalculatePositionsSparse() to ClusteringService — do NOT modify existing methods
- Do NOT change Channel-based batch processing in ImageScanner.cs
- English only in all code, comments, identifiers
- Preserve MVVM — no business logic in MainWindow.xaml.cs (console auto-scroll in code-behind is OK — it is purely presentation)
- All long operations on Task.Run — UI thread never blocked
- Use DynamicResource for all theme brushes
- LogService must be thread-safe
- Console panel must auto-scroll to latest line

## Done When

- [ ] Theme toggle switches dark/light instantly, persists via AppSettings.json
- [ ] Console panel shows live timestamped monospace log, auto-scrolls, is collapsible
- [ ] All operations (PCA, clustering, scan) are non-blocking — UI stays responsive
- [ ] Images appear progressively on Map canvas during PCA (batch-by-batch)
- [ ] Clusters tab shows empty state, "Compute clusters" button triggers lazy computation
- [ ] Sparse compression slider changes computation quality and logs the change
- [ ] GPU LED indicator shows correct status (green/amber/red)
- [ ] Cockpit panel expands to show all controls: GPU, threads, sparse, similarity, telemetry
- [ ] Status bar shows live: image count, cluster count, GPU info, DB size, theme toggle
- [ ] AppSettings.json round-trips all settings correctly
- [ ] Recalculate PCA button clears cache and reruns SVD

## Notes for Claude

### Theme ResourceDictionary swap
```csharp
var merged = Application.Current.Resources.MergedDictionaries;
var old = merged.FirstOrDefault(d => d.Source?.ToString().Contains("Theme") == true);
if (old != null) merged.Remove(old);
merged.Add(new ResourceDictionary {
    Source = new Uri("pack://application:,,,/Themes/DarkTheme.xaml")
});
```

### Console auto-scroll (code-behind OK here)
Subscribe to ConsoleLines.CollectionChanged in MainWindow.xaml.cs, call consoleScrollViewer.ScrollToBottom() — purely a view concern.

### GPU detection
```csharp
var providers = OrtEnv.Instance().GetAvailableProviders();
```
Wrap in try/catch. Run in Task.Run on startup to avoid blocking.

### CalculatePositionsSparse internal flow
1. Flatten all cluster images to vector list (same as CalculatePositions)
2. Compress: compressedVectors = allVectors.Select(v => ToSparse(v, sparseTopN)).ToList()
3. Report progress after compression step
4. Call existing ReduceTo2D_PCA(compressedVectors) — no algorithm change
5. Call existing NormalizePositions() — no algorithm change
6. Report completion
7. Build and return ClusterPosition list same as CalculatePositions

### LiteDB UpdateMany
```csharp
_collection.UpdateMany(v => { v.PcaX = null; v.PcaY = null; return v; }, _ => true);
```

### Files to create or modify

| Action | File |
|--------|------|
| CREATE | Themes/LightTheme.xaml |
| CREATE | Themes/DarkTheme.xaml |
| CREATE | Services/ThemeService.cs |
| CREATE | Services/LogService.cs |
| CREATE | Utility/GpuDetector.cs |
| MODIFY | Services/ClusteringService.cs — ADD ToSparse() + CalculatePositionsSparse() |
| MODIFY | Models/IVectorDatabase.cs — ADD ClearPcaCacheAsync() |
| MODIFY | Services/LiteDbVectorStore.cs — IMPLEMENT ClearPcaCacheAsync() |
| MODIFY | ViewModels/MainViewModel.cs — all new properties + commands + log wiring |
| MODIFY | App.xaml — MergedDictionaries for theme |
| MODIFY | App.xaml.cs — register ThemeService, LogService; load settings |
| MODIFY | MainWindow.xaml — cockpit panel, console panel, tabs, status bar |
