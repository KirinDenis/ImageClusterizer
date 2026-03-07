# SKILL-003: UX Rendering, Themes and Non-Blocking UI

## Goal
Improve the user experience significantly:
- Add dark/light theme toggle (persistent between sessions)
- Make PCA scatter view render progressively (images appear one by one as PCA is computed)
- Prevent UI freezing during clustering and PCA computation
- Reorganize tabs: PCA scatter is the primary view, Cosine Similarity clustering is secondary with an explicit "Compute clusters" button
- Add status bar at the bottom with live statistics

## Depends On
- SKILL-001 must be completed
- SKILL-002 must be completed (thumbnails and PCA cache must exist)

## Scope
- Work only inside: `ImageClusterizer/ImageClusterizer_WPF/`
- Do NOT touch: `Polygon/`, `ClusteringService.cs` algorithms

## Stack
- .NET 8, WPF (net8.0-windows10.0.17763.0)
- CommunityToolkit.Mvvm 8.4.0
- WPF ResourceDictionary for themes (no third-party theme libraries)

---

## What to implement

### 1. Dark / Light theme toggle

Create two ResourceDictionaries:
- `Themes/LightTheme.xaml`
- `Themes/DarkTheme.xaml`

Each must define the following named brushes used throughout the app:
```
AppBackground, AppSurface, AppBorder,
AppText, AppTextSecondary,
AppAccent, AppAccentHover,
ToolBarBackground, StatusBarBackground
```

In `App.xaml`: load `LightTheme.xaml` by default via MergedDictionaries.

Add `ThemeService` (new singleton service):
- `void ToggleTheme()` — swaps the active theme ResourceDictionary at runtime
- `Theme CurrentTheme { get; }` — enum: Light / Dark
- `void SaveThemePreference()` / `void LoadThemePreference()` — persist to `AppSettings.json` next to executable
  (use System.Text.Json, no extra NuGet packages)

Add toggle button to toolbar:
- Icon: ☀ (light) / 🌙 (dark) — switches based on current theme
- Tooltip: "Switch to Dark/Light theme"
- Bound to `ViewModel.ToggleThemeCommand`

Apply theme brushes throughout `MainWindow.xaml`:
- Window Background = AppBackground
- ToolBar Background = ToolBarBackground
- StatusBar Background = StatusBarBackground
- All text = AppText / AppTextSecondary
- Borders = AppBorder
- Cluster ellipses = AppAccent

### 2. Tab reorganization

Rename and reorder tabs:

**Tab 1: "Map" (was "Stochastic")** — PCA scatter view, this is the primary view
- Opens by default on startup
- Shows cached PCA positions immediately (from SKILL-002)
- Has its own "Recalculate PCA" button inside the tab (not in main toolbar)
  - Enabled only when not scanning
  - Triggers PCA recompute for all vectors and re-saves to DB

**Tab 2: "Clusters"** — Cosine Similarity cluster grid
- Does NOT auto-compute on load
- Shows empty state message: "Click 'Compute clusters' to group similar images"
- Has a "Compute clusters" button inside the tab header area or top of content
  - Shows a progress indicator (indeterminate ProgressBar) while computing
  - Bound to `ViewModel.ComputeClustersCommand`
  - Computes on background thread — UI must NOT freeze

### 3. Non-blocking PCA computation

Current problem: `ClusteringService.CalculatePositions` runs SVD on all vectors at once — this blocks UI for large collections.

Solution:
- Move PCA computation entirely to `Task.Run` with progress reporting via `IProgress<int>`
- Add `[ObservableProperty] private int pcaProgress` to `MainViewModel` (0-100)
- Add `[ObservableProperty] private bool isPcaComputing` 
- Show indeterminate ProgressBar in the Map tab while computing
- After PCA is done, populate `ImageItems` on UI thread using `Application.Current.Dispatcher.InvokeAsync`

### 4. Progressive scatter rendering

After PCA computation, instead of populating all `ImageItems` at once (which freezes WPF for 10k+ items):
- Add items to `ImageItems` in batches of 100 with a short `await Task.Delay(1)` between batches
- This keeps UI responsive and gives visual effect of items appearing progressively
- Use `Application.Current.Dispatcher.InvokeAsync` for each batch

### 5. Non-blocking cluster computation

`ClusterBySimilarity` is O(n²) — for 10k images this is 100M comparisons.

Solution:
- Run entirely in `Task.Run`
- Add `[ObservableProperty] private bool isClusterComputing` to `MainViewModel`
- Show indeterminate ProgressBar in Clusters tab while computing
- "Compute clusters" button disabled while computing
- After completion, populate `Clusters` collection on UI thread

### 6. Status bar

Add a status bar at the bottom of the window (`DockPanel` with `DockPanel.Dock="Bottom"`):

Left side:
- "📊 {VectorCount} images indexed"
- "🗂 {ClusterCount} clusters"
- "📐 Vector: {SelectedVectorType}"

Right side:
- "💾 DB: {DatabaseSizeText}" — formatted as KB/MB
- Current theme indicator (☀ or 🌙)

Add corresponding observable properties to `MainViewModel`:
`VectorCount`, `ClusterCount`, `DatabaseSizeText`
Update them after scan, after cluster compute, and after clear data.

---

## Constraints
- No third-party NuGet packages for themes — use WPF ResourceDictionary only
- Do NOT modify `ClusteringService.cs` algorithm internals
- Do NOT change Channel-based batch processing in `ImageScanner.cs`
- English only in all code, comments, identifiers
- Preserve MVVM structure — no code-behind logic in MainWindow.xaml.cs

## Done When
- Theme toggle button switches between dark and light mode instantly
- Theme preference is saved and restored on next app launch
- Map tab opens by default and shows PCA scatter immediately from cache (SKILL-002)
- "Recalculate PCA" button in Map tab works without freezing UI
- Images appear progressively in scatter view (batch rendering)
- Clusters tab shows empty state until user clicks "Compute clusters"
- "Compute clusters" runs without freezing UI, shows progress indicator
- Status bar shows correct counts and DB size
- UI remains responsive during all computation phases

## Notes for Claude
- WPF ResourceDictionary theme swap: remove old dict from MergedDictionaries, add new one
- For progressive rendering: `ObservableCollection` + batched `Dispatcher.InvokeAsync` is the correct pattern
- AppSettings.json: simple JSON with single property `{ "Theme": "Dark" }` — use `JsonSerializer`
- `DatabaseSizeText`: use `new FileInfo(StorageService.DatabasePath).Length` formatted as "1.2 MB"
- Theme-aware cluster ellipse color: bind Fill to `{DynamicResource AppAccent}` (DynamicResource updates at runtime, StaticResource does not)
