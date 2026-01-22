using ImageClusterizer.Services;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Xaml;
using System;


namespace ImageClusterizer
{
    public partial class App : Application
    {
        public static IServiceProvider? Services { get; private set; }

        public static Window? mainWindow { get; private set; }

        private readonly IHost host;
        public App()
        {
            InitializeComponent();

            host = Host.CreateDefaultBuilder()
                .ConfigureServices((context, services) =>
                {
                    ConfigureServices(services);
                })
                .Build();

            Services = host.Services;
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
                var dbPath = System.IO.Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "ImageClusterizer",
                    "vectors.db");

                // Создаём папку если нет
                System.IO.Directory.CreateDirectory(
                    System.IO.Path.GetDirectoryName(dbPath));

                return new LiteDbVectorStore(dbPath);
            });

            services.AddTransient<ImageScanner>();

            services.AddTransient<ClusteringService>();

            services.AddSingleton<MainViewModel>();

            services.AddSingleton<MainWindow>();
        }

        protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
        {
            mainWindow = Services?.GetRequiredService<MainWindow>();
            mainWindow?.Activate();
        }
    }
}
