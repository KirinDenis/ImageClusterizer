namespace ImageClusterizer.ViewModels;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Models;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using Microsoft.UI.Xaml;

using Microsoft.UI.Xaml.Data;
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

        //----
        

        [ObservableProperty]
        private ObservableCollection<ClusterVisualItem> clusterItems = new();

        [ObservableProperty]
        private ObservableCollection<ImageVisualItem> imageItems = new();

        [ObservableProperty]
        private double canvasWidth = 1000;

        [ObservableProperty]
        private double canvasHeight = 1000;

        [ObservableProperty]
        private double zoomLevel = 1.0;

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

        public void LoadClusters(List<ImageCluster> clusters)
        {
            ClusterItems.Clear();
            ImageItems.Clear();

            // Вычисляем позиции
            var positions = clusteringService.CalculatePositions(clusters,
                (int)CanvasWidth, (int)CanvasHeight);

            // Группируем по кластерам
            var grouped = positions.GroupBy(p => p.ClusterId);

            foreach (var group in grouped)
            {
                // Центроид кластера
                var centroid = group.FirstOrDefault(p => p.IsCentroid);
                if (centroid != null)
                {
                    ClusterItems.Add(new ClusterVisualItem
                    {
                        ClusterId = centroid.ClusterId,
                        X = centroid.X,
                        Y = centroid.Y,
                        ImageCount = group.Count(p => !p.IsCentroid)
                    });
                }

                // Изображения в кластере
                foreach (var imagePos in group.Where(p => !p.IsCentroid))
                {
                    ImageItems.Add(new ImageVisualItem
                    {
                        ClusterId = imagePos.ClusterId,
                        X = imagePos.X,
                        Y = imagePos.Y,
                        FilePath = imagePos.ImageVector.FilePath,
                        ThumbnailPath = imagePos.ImageVector.FilePath  // или thumbnail
                    });
                }
            }
        }

        [RelayCommand]
        private void ZoomIn()
        {
            ZoomLevel = Math.Min(ZoomLevel * 1.2, 10.0);
        }

        [RelayCommand]
        private void ZoomOut()
        {
            ZoomLevel = Math.Max(ZoomLevel / 1.2, 0.1);
        }

        [RelayCommand]
        private void ResetZoom()
        {
            ZoomLevel = 1.0;
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

        [RelayCommand]
        private void OpenClusterMap()
        {
          //  var mapWindow = new ClusterMapView();
          //  var viewModel = new ClusterMapViewModel();

            // Загружаем кластеры
            LoadClusters(Clusters.ToList());

            //mapWindow.ViewModel = viewModel;
            //mapWindow.Activate();
        }

        private async Task ClusterImagesAsync()
        {
            var vectors = await vectorDatabase.GetAllAsync();
            var clusters = await Task.Run(() => clusteringService.ClusterBySimilarity(vectors, 0.9f));

            Clusters.Clear();
            foreach (var cluster in clusters)
            {
                Clusters.Add(cluster);
            }
        }
    }

