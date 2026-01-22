using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Models;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using Microsoft.UI.Xaml;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Windows.Storage;

namespace ImageClusterizer.ViewModels
{
    public partial class MainViewModel: ObservableObject
    {
        private readonly ImageScanner imageScanner;
        private readonly IVectorDatabase vectorDatabase;
        private readonly ClusteringService clusteringService;


        [ObservableProperty]
        private ObservableCollection<ImageCluster> clusters = new();

        [ObservableProperty]
        private string currentFile = "";

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(ProcessedCountText))]
        private int processedCount;

        public string ProcessedCountText => ProcessedCount.ToString();

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(TotalCountText))]
        private int totalCount;

        public string TotalCountText => TotalCount.ToString();

        [ObservableProperty]
        private double progress;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(IsNotScanning), nameof(ProgressVisibility))]
        private bool isScanning;

        public bool IsNotScanning => !IsScanning;
        public Visibility ProgressVisibility => IsScanning ? Visibility.Visible : Visibility.Collapsed;

        private CancellationTokenSource? cts;
        public MainViewModel(ImageScanner imageScanner, IVectorDatabase vectorDatabase, ClusteringService clusteringService)
        {
            this.imageScanner = imageScanner;
            this.vectorDatabase = vectorDatabase;
            this.clusteringService = clusteringService;
        }

        [RelayCommand]
        private async Task StartScanImagesAsync()
        {
            StorageFolder folder = await Utility.SelectFolderDiagoAsync();

            if (folder == null)
            {
                return;
            }

            IsScanning = true;
            cts = new CancellationTokenSource();

            try
            {
                Clusters.Clear();

                await foreach (var progress in imageScanner.ScanFolderAsync(folder.Path, cts.Token))
                {
                    CurrentFile = Path.GetFileName(progress.CurrentFile);
                    ProcessedCount = progress.ProcessedCount;
                    TotalCount = progress.TotalCount;
                    Progress = (double)ProcessedCount / TotalCount * 100;
                }

                await ClusterImagesAsync();
            }
            finally
            {
                IsScanning = false;
                cts?.Cancel();
                cts?.Dispose();                
            }
        }

        [RelayCommand]
        private async Task LoadExistingClustersAsync()
        {
            await ClusterImagesAsync();
        }

        [RelayCommand]
        private void CancelScan()
        {
            cts?.Cancel();
        }

        private async Task ClusterImagesAsync()
        {
            var vectors = await vectorDatabase.GetAllAsync();
            var clusters = await Task.Run(() => clusteringService.ClusterBySimilarity(vectors, 0.25f));

            Clusters.Clear();
            foreach (var cluster in clusters)
            {
                Clusters.Add(cluster);
            }
        }
    }
}
