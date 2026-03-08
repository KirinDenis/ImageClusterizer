namespace ImageClusterizer;

using ImageClusterizer.Services;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using System.Windows;

public partial class App : Application
{
    public static IServiceProvider? Services { get; private set; }
    public static Window? mainWindow { get; private set; }

    private readonly IHost host;

    public App()
    {
        try
        {
            DispatcherUnhandledException += (s, e) =>
            {
#if DEBUG
                e.Handled = false;
#else
                e.Handled = true;
#endif
                File.WriteAllText(
                    Path.Combine(AppContext.BaseDirectory, "crash.log"),
                    e.Exception.ToString());
                System.Diagnostics.Debug.WriteLine(e.Exception.ToString());
            };

            host = Host.CreateDefaultBuilder()
                .ConfigureServices((context, services) => ConfigureServices(services))
                .Build();

            Services = host.Services;
        }
        catch (Exception ex)
        {
            File.WriteAllText(
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "crash.log"),
                ex.ToString());
            throw;
        }
    }

    private void ConfigureServices(IServiceCollection services)
    {
        // Theme and log services — registered before ViewModel so they can be injected
        services.AddSingleton<ThemeService>();
        services.AddSingleton<LogService>();

        // StorageService owns all path resolution (data/vectors.db, thumbnails/)
        services.AddSingleton<StorageService>();

        // ResNet50 vectorizer
        services.AddSingleton<IVectorService>(sp =>
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "resnet50-v2-7.onnx");
            return new ResNetVectorizer(modelPath);
        });

        // LiteDB store — database path comes from StorageService
        services.AddSingleton<IVectorDatabase>(sp =>
        {
            var storage = sp.GetRequiredService<StorageService>();
            return new LiteDbVectorStore(storage.DatabasePath);
        });

        services.AddTransient<ImageScanner>();
        services.AddTransient<ClusteringService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<MainWindow>();
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        try
        {
            // Apply saved theme before window is shown
            var themeService = Services?.GetRequiredService<ThemeService>();
            themeService?.LoadPreference();

            mainWindow = Services?.GetRequiredService<MainWindow>();
            mainWindow?.Show();
        }
        catch (Exception ex)
        {
            File.WriteAllText(
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "crash.log"),
                ex.ToString());
            Shutdown();
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        host.Dispose();
        base.OnExit(e);
    }
}
