# SKILL-001: Core Pipeline Stabilization

## Goal
Fix and stabilize the existing core pipeline in the WPF project (ImageClusterizer_WPF).
Remove dead WinUI3 code and artifacts. Integrate proper vector type selection (Embedding vs Logit).
The result must be a clean, buildable .NET 8 WPF project with a working end-to-end pipeline.

## Scope
- Work only inside: ImageClusterizer/ImageClusterizer_WPF/
- Ignore: Polygon/ folder (experimental sandbox, do not read or modify)
- Note: ImageClusterizer/ImageClusterizer/ (old WinUI3 project) will be deleted by developer

## Stack
- .NET 8, WPF (net8.0-windows10.0.17763.0)
- CommunityToolkit.Mvvm 8.4.0
- Microsoft.ML.OnnxRuntime (ResNet50-v2-7.onnx)
- LiteDB 5.0.21
- MathNet.Numerics 5.0.0
- SixLabors.ImageSharp 3.1.12
- Microsoft.Extensions.DependencyInjection / Hosting

## What to fix

### 1. Remove WinUI3 references from Models.cs
- Remove all Microsoft.UI.Xaml using statements
- Remove commented-out PositionToMarginConverter (WinUI3 artifact)
- Replace with WPF-compatible equivalents if needed (using System.Windows;)
- Keep all models in one file for now, just clean it up

### 2. Add VectorType support
- Add enum: public enum VectorType { Embedding, Logit }
- Add VectorType property to ImageVector model
- Add VectorType field to ImageVectorEntity in LiteDbVectorStore.cs
- Update ResNetVectorizer to accept VectorType parameter:
  - Logit: raw output of last layer (1000D)
  - Embedding: output of penultimate layer (2048D)
- Add SelectedVectorType observable property to MainViewModel
- Pass SelectedVectorType to ResNetVectorizer before scan starts

### 3. Fix FileSize persistence
- Add FileSize property to ImageVectorEntity
- Map FileSize in SaveAsync and GetAllAsync in LiteDbVectorStore.cs

### 4. Fix IVectorService interface
- Ensure GetEmbeddingAsync signature matches ResNetVectorizer implementation
- Add VectorType parameter support

### 5. General code cleanup
- All comments and identifiers must be in English only
- Remove any leftover WinUI3 or WinRT using statements project-wide
- Ensure App.xaml.cs uses WPF Application base, not WinUI3

## Constraints
- Do NOT change the Channel-based batch processing logic in ImageScanner.cs
- Do NOT change clustering algorithms in ClusteringService.cs
- Do NOT touch Polygon/ folder
- English only in all code, comments, and identifiers
- Preserve existing MVVM structure

## Done When
- Project builds with 0 errors related to WinUI3
- User can select VectorType (Embedding or Logit) in the UI before starting scan
- FileSize is correctly saved and loaded from LiteDB
- No Microsoft.UI.Xaml references remain in ImageClusterizer_WPF

## Notes for Claude
- Read current file contents before modifying - do not overwrite working code
- When modifying ResNetVectorizer, check available ONNX output node names at runtime if unsure
- The ONNX model file is resnet50-v2-7.onnx (already referenced in .csproj)
- ImageScanner uses IVectorService.GetEmbeddingAsync - update interface carefully
