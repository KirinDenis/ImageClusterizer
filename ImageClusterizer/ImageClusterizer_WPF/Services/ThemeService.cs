namespace ImageClusterizer.Services;

using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;

/// <summary>
/// Manages application theme (Light/Dark) with runtime switching and persistence.
/// Swaps WPF ResourceDictionary at runtime so all DynamicResource bindings update instantly.
/// Settings are persisted to AppSettings.json next to the executable.
/// </summary>
public class ThemeService
{
    public enum Theme { Light, Dark }

    private const string LightThemeUri = "pack://application:,,,/Themes/LightTheme.xaml";
    private const string DarkThemeUri  = "pack://application:,,,/Themes/DarkTheme.xaml";

    public Theme CurrentTheme { get; private set; } = Theme.Light;

    /// <summary>Toggles between Light and Dark, saves preference</summary>
    public void ToggleTheme()
    {
        ApplyTheme(CurrentTheme == Theme.Light ? Theme.Dark : Theme.Light);
    }

    /// <summary>Applies the specified theme by swapping the theme ResourceDictionary</summary>
    public void ApplyTheme(Theme theme)
    {
        CurrentTheme = theme;

        var merged = Application.Current.Resources.MergedDictionaries;

        // Remove the currently active theme dictionary (if any)
        var old = merged.FirstOrDefault(d =>
            d.Source != null &&
            (d.Source.ToString().Contains("LightTheme") ||
             d.Source.ToString().Contains("DarkTheme")));

        if (old != null)
            merged.Remove(old);

        // Add the new theme dictionary — all DynamicResource bindings update automatically
        var uri = theme == Theme.Dark ? DarkThemeUri : LightThemeUri;
        merged.Add(new ResourceDictionary { Source = new Uri(uri) });

        SavePreference();
    }

    /// <summary>Saves current theme preference to AppSettings.json</summary>
    public void SavePreference()
    {
        try
        {
            var settings = AppSettings.Load();
            settings.Theme = CurrentTheme.ToString();
            AppSettings.Save(settings);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"ThemeService.SavePreference failed: {ex.Message}");
        }
    }

    /// <summary>Loads and applies saved theme preference from AppSettings.json</summary>
    public void LoadPreference()
    {
        try
        {
            var settings = AppSettings.Load();
            var theme = Enum.TryParse<Theme>(settings.Theme, out var t) ? t : Theme.Light;
            ApplyTheme(theme);
        }
        catch
        {
            ApplyTheme(Theme.Light);
        }
    }
}

/// <summary>
/// Application settings model — persisted to AppSettings.json next to the executable.
/// All settings are optional with sensible defaults.
/// </summary>
public class AppSettings
{
    private static readonly string SettingsPath =
        Path.Combine(AppContext.BaseDirectory, "AppSettings.json");

    private static readonly JsonSerializerOptions JsonOptions =
        new() { WriteIndented = true };

    public string Theme              { get; set; } = "Light";
    public int    SparseTopN         { get; set; } = 2048;
    public bool   UseGpu             { get; set; } = true;
    public int    ThreadCount        { get; set; } = 0;   // 0 = auto (ProcessorCount)
    public double SimilarityThreshold{ get; set; } = 0.85;
    public bool   IsConsoleExpanded  { get; set; } = true;

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
            }
        }
        catch { /* return defaults on any error */ }
        return new AppSettings();
    }

    public static void Save(AppSettings settings)
    {
        try
        {
            var json = JsonSerializer.Serialize(settings, JsonOptions);
            File.WriteAllText(SettingsPath, json);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"AppSettings.Save failed: {ex.Message}");
        }
    }
}
