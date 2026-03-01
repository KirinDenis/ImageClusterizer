using ImageClusterizer.Services;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Xaml;
using System;
using System.IO;


namespace ImageClusterizer
{
    public partial class App : Application
    {
        public static IServiceProvider? Services { get; private set; }

        public static Window? mainWindow { get; private set; }

        private readonly IHost host;
        public App()
        {
            try
            {
                InitializeComponent();

                UnhandledException += (s, e) =>
                {
                    e.Handled = true;
                    File.WriteAllText(
                        Path.Combine(AppContext.BaseDirectory, "crash.log"),
                        e.Exception.ToString()
                    );

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
                var modelPath = System.IO.Path.Combine(
                    AppContext.BaseDirectory,
                    "resnet50-v2-7.onnx");
                return new ResNetVectorizer(modelPath);
            });

            services.AddSingleton<IVectorDatabase>(sp =>
            {
                //var dbPath = System.IO.Path.Combine(
                //  Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                //"ImageClusterizer",
                //"vectors.db");

                //System.IO.Directory.CreateDirectory(
                //System.IO.Path.GetDirectoryName(dbPath));

                var dbPath = System.IO.Path.Combine(
                    AppContext.BaseDirectory,
                    "vectors.db");


                return new LiteDbVectorStore(dbPath);
            });

            services.AddTransient<ImageScanner>();

            services.AddTransient<ClusteringService>();

            services.AddSingleton<MainViewModel>();

            services.AddSingleton<MainWindow>();
        }

        protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
        {
            try
            {
                mainWindow = Services?.GetRequiredService<MainWindow>();
                mainWindow?.Activate();
            }
            catch (Exception ex)
            {
                File.WriteAllText(
                    Path.Combine(Environment.GetFolderPath(
                        Environment.SpecialFolder.Desktop), "crash.log"),
                    ex.ToString());
            }
        }
    }
}
