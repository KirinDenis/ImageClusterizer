# SKILL-002: Thumbnails, PCA Cache, Storage Layout and Data Cleanup

## Goal
Introduce persistent storage for image thumbnails and PCA coordinates.
Reorganize data storage into separate folders.
Add a "Clear all data" action so users can remove all locally stored copies of their image data.
After this SKILL the app must show PCA scatter view instantly on startup (from cache),
without recomputing anything, unless the user explicitly requests a recalculation.

## Depends On
- SKILL-001 must be completed and building

## Scope
- Work only inside: `ImageClusterizer/ImageClusterizer_WPF/`
- Do NOT touch: `Polygon/`, `ClusteringService.cs` algorithms

## Stack
- .NET 8, WPF (net8.0-windows10.0.17763.0)
- LiteDB 5.0.21
- SixLabors.ImageSharp 3.1.12 (already used for preprocessing — reuse for thumbnail save)
- CommunityToolkit.Mvvm 8.4.0

---

## Storage Layout

Introduce a `StorageService` (new file) that owns all path resolution.
All data lives under a base folder next to the executable (or AppData — see note below).

```
<AppBaseDirectory>/
  data/
    vectors.db          <- LiteDB database (move from root)
  thumbnails/
    <sha256_of_filepath>.jpg   <- 224x224 JPEG thumbnails
```

### StorageService responsibilities
- `string DatabasePath` — path to vectors.db
- `string ThumbnailsFolder` — path to thumbnails/ folder (ensure it exists)
- `string GetThumbnailPath(string imageFilePath)` — returns thumbnail path using SHA256 hash of original path as filename (avoids path separator issues)
- `Task ClearAllDataAsync()` — deletes vectors.db and all files in thumbnails/ folder

### Update App.xaml.cs
- Register `StorageService` as singleton
- Pass `StorageService.DatabasePath` to `LiteDbVectorStore`
- Pass `StorageService.ThumbnailsFolder` where needed

---

## What to implement

### 1. Thumbnail generation during scan
In `ImageScanner.cs`, after extracting the vector:
- Save the already-preprocessed 224x224 image as a JPEG to `StorageService.GetThumbnailPath(imageFile)`
- Use SixLabors.ImageSharp JpegEncoder with quality 85
- Skip saving if thumbnail already exists (same logic as DB check)
- Store thumbnail path in `ImageVector.ThumbnailPath` (add this field to the model)

### 2. Extend ImageVector and ImageVectorEntity models
Add to `ImageVector` record:
```csharp
public string? ThumbnailPath { get; init; }  // path to 224x224 thumbnail file
public float? PcaX { get; init; }            // cached 2D PCA X coordinate (null = not yet computed)
public float? PcaY { get; init; }            // cached 2D PCA Y coordinate (null = not yet computed)
```

Add same fields to `ImageVectorEntity` and update SaveAsync / GetAllAsync mapping.

### 3. PCA coordinate caching
Add method to `IVectorDatabase`:
```csharp
Task SavePcaCoordinatesAsync(string filePath, float pcaX, float pcaY);
```

After PCA is computed in `ClusteringService.CalculatePositions`:
- For each image, call `SavePcaCoordinatesAsync` to persist its 2D position
- On next startup, if ALL vectors have PcaX/PcaY set, skip PCA recomputation entirely
- If ANY vector is missing PcaX/PcaY (new images added), recompute PCA for all and re-save

### 4. Fast startup display
In `MainViewModel`:
- On app start (or when user clicks "Reload from database"), call `GetAllAsync()`
- If all returned vectors have PcaX/PcaY != null — populate `ImageItems` directly from cache, skip `ClusteringService.CalculatePositions`
- If PCA cache is incomplete — compute PCA, save results, then display

### 5. Thumbnail display
- `ImageVisualItem.ThumbnailPath` should now point to the local thumbnail file (not the original image)
- This makes display fast even for files that have moved or are on slow network drives
- In `MainWindow.xaml`, the `ImageBrush` and tooltip `Image` already bind to `ThumbnailPath` — no XAML change needed if the path is correct

### 6. Clear all data action
Add `[RelayCommand] private async Task ClearAllDataAsync()` to `MainViewModel`:
- Show a WPF `MessageBox` confirmation dialog:
  Title: "Clear all data"
  Message: "This will permanently delete all stored vectors, thumbnails, and cached positions. Your original image files will NOT be affected. Continue?"
  Buttons: Yes / No
- If confirmed: call `StorageService.ClearAllDataAsync()`, then clear all observable collections
- Add a toolbar button with 🗑 icon bound to this command, enabled only when `IsNotScanning`

---

## Constraints
- Do NOT modify clustering algorithms in `ClusteringService.cs`
- Do NOT change Channel-based batch processing structure in `ImageScanner.cs`
- ThumbnailPath in ImageVisualItem already exists — just ensure it is populated correctly
- English only in all code, comments, identifiers
- Preserve MVVM structure

## Done When
- App saves 224x224 thumbnails to `data/thumbnails/` during scan
- App saves PCA coordinates to LiteDB after each PCA computation
- On reload, if PCA cache is complete — scatter view appears instantly (no recompute)
- "Clear all data" button with confirmation dialog works and deletes both DB and thumbnails
- `vectors.db` lives in `data/` subfolder, not in root next to executable

## Notes for Claude
- SHA256 of the original file path (not file content) is used as thumbnail filename — consistent and fast
- ImageScanner already preprocesses to 224x224 for ONNX — reuse that Image<Rgb24> object to save thumbnail before disposing
- PCA recompute trigger: `vectors.Any(v => v.PcaX == null)`
- Do NOT store absolute paths in thumbnails folder — store only the filename (hash) and reconstruct full path via StorageService
