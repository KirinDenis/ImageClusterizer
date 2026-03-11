# Contributing to Image Clusterizer

Thank you for your interest in contributing to **Image Clusterizer**! This project is a Windows desktop AI application built on .NET 8 WPF, ONNX Runtime, and ResNet-50. Contributions of all kinds are welcome — bug fixes, new features, algorithm improvements, documentation, and Polygon research prototypes.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Submitting a Pull Request](#submitting-a-pull-request)

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/ImageClusterizer.git
   cd ImageClusterizer
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```

## Development Setup

**Requirements:**
- Windows 10 / 11 (x64)
- .NET 8 SDK
- Visual Studio 2022+ or Rider (recommended)
- Optional: NVIDIA GPU with CUDA 11+ for inference testing

**Download the ONNX model (first-time only):**
```bash
curl -L -o ImageClusterizer_WPF/resnet50-v2-7.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
```

**Build and run:**
```bash
dotnet build
dotnet run --project ImageClusterizer/ImageClusterizer_WPF
```

## Project Structure

```
ImageClusterizer/
├── ImageClusterizer_WPF/   # Main WPF application (MVVM)
├── Polygon/                # Standalone research & prototype console projects
│   ├── 1_ResNet50_GetLogits_Test/
│   ├── 2_ResNet50_GetEmbedding_Test/
│   ├── 3_ResNet50_Image_similarity_search_test/
│   ├── 4_ResNet50_2D_PCA_test/
│   └── 5_ResNet50_Sparse_Dot_Product_test/
```

The **Polygon** projects are a great place to experiment with algorithms before integrating them into the main app.

## How to Contribute

### Reporting Bugs

Please use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) issue template and include:
- OS version and .NET runtime version
- GPU model and whether CUDA/DirectML was enabled
- Steps to reproduce, expected vs. actual behavior
- Approximate image collection size if relevant

### Suggesting Features

Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template. Describe the use case clearly — especially how it fits into the vectorization, PCA, or clustering pipeline.

### Improving the Polygon Prototypes

The `Polygon/` folder welcomes new standalone research projects that validate new algorithms (e.g., alternative dimensionality reduction methods, different CNN backbones, clustering strategies). Each project should be self-contained and include a brief README comment explaining what it demonstrates.

### Fixing Bugs or Implementing Features

1. Check open [issues](https://github.com/KirinDenis/ImageClusterizer/issues) or create one first.
2. Keep changes focused — one feature or fix per PR.
3. Test with a representative image collection (100–10,000 images recommended).
4. If your change affects GPU inference or PCA performance, include benchmark numbers.

## Coding Guidelines

- Follow **MVVM** architecture for any WPF changes (use CommunityToolkit.Mvvm bindings).
- Use C# naming conventions (PascalCase for public members, camelCase for locals).
- Keep heavy computation (ONNX inference, SVD) off the UI thread — use `Task.Run` or async patterns.
- Prefer adding new Analysis Profiles rather than changing defaults.
- All data must remain local — no network calls, no telemetry, no cloud uploads.
- Comment non-obvious math (e.g., Halko RSVD parameters, cosine similarity thresholds).

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`.
2. Open a PR using the pull request template.
3. Describe what you changed and why.
4. Link to the related issue (if any).
5. A maintainer will review and provide feedback within a reasonable time.

We appreciate every contribution, no matter how small. Thank you for helping make Image Clusterizer better!
