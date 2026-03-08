namespace ImageClusterizer.Utlility;

using Microsoft.ML.OnnxRuntime;
using System;
using System.Linq;

/// <summary>
/// Detects GPU availability via OnnxRuntime execution providers.
/// Supports CUDA (NVIDIA) and DirectML (AMD/Intel on Windows).
/// Detection is non-blocking when run via Task.Run.
/// </summary>
public static class GpuDetector
{
    public record GpuInfo(
        bool IsAvailable,
        string ProviderName,
        string DeviceName);

    /// <summary>
    /// Detects the best available execution provider.
    /// Returns CUDA > DirectML > CPU in priority order.
    /// Wraps OrtEnv call in try/catch — safe to call even if no GPU driver installed.
    /// </summary>
    public static GpuInfo Detect()
    {
        try
        {
            var providers = OrtEnv.Instance().GetAvailableProviders();

            if (providers.Contains("CUDAExecutionProvider"))
                return new GpuInfo(true, "CUDA", "CUDA GPU (NVIDIA)");

            if (providers.Contains("DmlExecutionProvider"))
                return new GpuInfo(true, "DirectML", "GPU (DirectML — AMD/Intel)");

            if (providers.Contains("ROCMExecutionProvider"))
                return new GpuInfo(true, "ROCM", "GPU (ROCm — AMD)");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"GpuDetector.Detect failed: {ex.Message}");
        }

        return new GpuInfo(false, "CPU", "No GPU detected");
    }

    /// <summary>
    /// Returns a short status string for display in the cockpit panel.
    /// Examples: "CUDA GPU (NVIDIA)", "GPU (DirectML)", "No GPU detected"
    /// </summary>
    public static string GetStatusText(GpuInfo info, bool useGpu)
    {
        if (!info.IsAvailable) return "CPU only";
        if (!useGpu) return $"{info.DeviceName} (disabled)";
        return info.DeviceName;
    }
}
