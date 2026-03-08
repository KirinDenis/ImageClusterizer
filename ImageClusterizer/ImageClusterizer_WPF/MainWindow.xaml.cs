using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System.Collections.Specialized;
using System.Windows;
using System.Windows.Controls;

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

            // Subscribe to ConsoleLines collection changes for auto-scroll
            // This is a pure view concern — code-behind is acceptable here
            ViewModel.ConsoleLines.CollectionChanged += OnConsoleLinesChanged;
        }

        private void OnConsoleLinesChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            // Auto-scroll console to show latest line
            if (ConsoleScrollViewer != null)
            {
                ConsoleScrollViewer.ScrollToBottom();
            }
        }
    }
}
