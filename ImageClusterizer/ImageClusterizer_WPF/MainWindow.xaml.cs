using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System.Windows;

namespace ImageClusterizer
{
    public partial class MainWindow : Window
    {
        /// <summary>
        /// The name must be ViewModel for XAML binding
        /// </summary>
        public MainViewModel ViewModel { get; private set; }

        public MainWindow()
        {
            InitializeComponent();

            ViewModel = (App.Services?.GetRequiredService<MainViewModel>())!;
            DataContext = this;
        }
    }
}
