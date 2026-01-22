using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageClusterizer.Utlility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;

namespace ImageClusterizer.ViewModels
{
    public partial class MainViewModel: ObservableObject
    {

        [RelayCommand]
        private async Task StartScanImagesAsync()
        {
            StorageFolder folder = await Utility.SelectFolderDiagoAsync();

            if (folder == null)
            {
                return;
            }


        }
    }
}
