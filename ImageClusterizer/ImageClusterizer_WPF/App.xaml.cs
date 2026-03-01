namespace ImageClusterizer;

using ImageClusterizer.Services;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
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
                .ConfigureServices((context, services) =>
                {
                    ConfigureServices(services);
                })
                .Build();

            Services = host.Services;
        }
        catch (Exception ex)
        {
            File.WriteAllText(
                Path.Combine(Environment.GetFolderPath(
                    Environment.SpecialFolder.Desktop), "crash.log"),
                ex.ToString());
            throw;
        }
    }

    private void ConfigureServices(IServiceCollection services)
    {
        services.AddSingleton<IVectorService>(sp =>
        {
            var modelPath = Path.Combine(
                AppContext.BaseDirectory,
                "resnet50-v2-7.onnx");
            return new ResNetVectorizer(modelPath);
        });

        services.AddSingleton<IVectorDatabase>(sp =>
        {
            var dbPath = Path.Combine(
                AppContext.BaseDirectory,
                "vectors.db");
            return new LiteDbVectorStore(dbPath);
        });

        services.AddTransient<ImageScanner>();
        services.AddTransient<ClusteringService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<MainWindow>();
    }

    // OnLaunched → OnStartup в WPF
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        try
        {
            mainWindow = Services?.GetRequiredService<MainWindow>();
            mainWindow?.Show();  // Show() вместо Activate()
        }
        catch (Exception ex)
        {
            File.WriteAllText(
                Path.Combine(Environment.GetFolderPath(
                    Environment.SpecialFolder.Desktop), "crash.log"),
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
