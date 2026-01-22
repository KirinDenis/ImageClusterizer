using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Services;
using ImageClusterizer.Utlility;
using System;
using System.Collections.Generic;
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

        private CancellationTokenSource? cts;
        public MainViewModel(ImageScanner imageScanner)
        {
            this.imageScanner = imageScanner;
        }

        [RelayCommand]
        private async Task StartScanImagesAsync()
        {
            StorageFolder folder = await Utility.SelectFolderDiagoAsync();

            if (folder == null)
            {
                return;
            }

            cts = new CancellationTokenSource();

            await  foreach(var progress in  imageScanner.ScanFolderAsync(folder.Path, cts.Token))
            {
                //TODO: show progress at XAML here
            };


        }
    }
}
