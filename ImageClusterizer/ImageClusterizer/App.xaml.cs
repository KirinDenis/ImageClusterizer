using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using ImageClusterizer.Services;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using Microsoft.UI.Xaml.Shapes;
using Windows.ApplicationModel;
using Windows.ApplicationModel.Activation;
using Windows.Foundation;
using Windows.Foundation.Collections;


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
            services.AddTransient<ImageScanner>();

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
