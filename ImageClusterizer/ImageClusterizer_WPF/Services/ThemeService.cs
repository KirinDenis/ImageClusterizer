namespace ImageClusterizer.Services;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;

/// <summary>
/// Manages application theme (Light/Dark) with runtime switching and persistence.
/// </summary>
public class ThemeService
{
    public enum Theme { Light, Dark }

    private const string LightThemeUri = "pack://application:,,,/Themes/LightTheme.xaml";
    private const string DarkThemeUri  = "pack://application:,,,/Themes/DarkTheme.xaml";

    public Theme CurrentTheme { get; private set; } = Theme.Light;

    public void ToggleTheme() => ApplyTheme(CurrentTheme == Theme.Light ? Theme.Dark : Theme.Light);

    public void ApplyTheme(Theme theme)
    {
        CurrentTheme = theme;
        var merged = Application.Current.Resources.MergedDictionaries;
        var old = merged.FirstOrDefault(d => d.Source != null &&
            (d.Source.ToString().Contains("LightTheme") || d.Source.ToString().Contains("DarkTheme")));
        if (old != null) merged.Remove(old);
        merged.Add(new ResourceDictionary { Source = new Uri(theme == Theme.Dark ? DarkThemeUri : LightThemeUri) });
        SavePreference();
    }

    public void SavePreference()
    {
        try { var s = AppSettings.Load(); s.Theme = CurrentTheme.ToString(); AppSettings.Save(s); }
        catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"ThemeService.SavePreference: {ex.Message}"); }
    }

    public void LoadPreference()
    {
        try { var s = AppSettings.Load(); ApplyTheme(Enum.TryParse<Theme>(s.Theme, out var t) ? t : Theme.Light); }
        catch { ApplyTheme(Theme.Light); }
    }
}

// ============================================================================
// Analysis Profile
// ============================================================================
/// <summary>
/// Named preset of analysis parameters. Users create profiles to quickly switch
/// between different analysis configurations.
/// </summary>
public class AnalysisProfile
{
    public string Name { get; set; } = "Default";
    public int SparseTopN { get; set; } = 2048;
    public double SimilarityThreshold { get; set; } = 0.85;
    public string VectorType { get; set; } = "Embedding";
    public bool UseGpu { get; set; } = true;
}

// ============================================================================
// AppSettings
// ============================================================================
/// <summary>
/// Application settings model — persisted to AppSettings.json next to the executable.
/// </summary>
public class AppSettings
{
    private static readonly string SettingsPath =
        Path.Combine(AppContext.BaseDirectory, "AppSettings.json");
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public string Theme { get; set; } = "Light";
    public int SparseTopN { get; set; } = 2048;
    public bool UseGpu { get; set; } = true;
    public int ThreadCount { get; set; } = 0;
    public double SimilarityThreshold { get; set; } = 0.85;
    public bool IsConsoleExpanded { get; set; } = true;
    public List<AnalysisProfile> Profiles { get; set; } = GetDefaultProfiles();
    public string LastUsedProfile { get; set; } = "Default";

    private static List<AnalysisProfile> GetDefaultProfiles() => new()
    {
        new AnalysisProfile
        {
            Name = "Default",
            SparseTopN = 2048,
            SimilarityThreshold = 0.85,
            VectorType = "Embedding",
            UseGpu = true
        },
        new AnalysisProfile
        {
            Name = "Fast (compressed)",
            SparseTopN = 256,
            SimilarityThreshold = 0.80,
            VectorType = "Embedding",
            UseGpu = true
        },
        new AnalysisProfile
        {
            Name = "Strict (deduplication)",
            SparseTopN = 2048,
            SimilarityThreshold = 0.97,
            VectorType = "Embedding",
            UseGpu = true
        },
        new AnalysisProfile
        {
            Name = "Logit classes",
            SparseTopN = 1000,
            SimilarityThreshold = 0.75,
            VectorType = "Logit",
            UseGpu = true
        }
    };

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                var s = JsonSerializer.Deserialize<AppSettings>(json);
                if (s != null)
                {
                    // Ensure default profiles exist if not persisted
                    if (s.Profiles == null || s.Profiles.Count == 0)
                        s.Profiles = GetDefaultProfiles();
                    return s;
                }
            }
        }
        catch { }
        return new AppSettings();
    }

    public static void Save(AppSettings settings)
    {
        try { File.WriteAllText(SettingsPath, JsonSerializer.Serialize(settings, JsonOptions)); }
        catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"AppSettings.Save: {ex.Message}"); }
    }
}
