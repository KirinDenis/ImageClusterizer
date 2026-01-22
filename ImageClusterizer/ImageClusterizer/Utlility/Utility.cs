using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace ImageClusterizer.Utlility
{
    public static class Utility
    {
        public static async Task<StorageFolder> SelectFolderDiagoAsync()
        {
            FolderPicker folderPicker = new();

            //WinUI3 -> need this for modal openning Folder Picker
            nint hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.mainWindow);
            WinRT.Interop.InitializeWithWindow.Initialize(folderPicker, hwnd);

            folderPicker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary;

            //for folder picker we can't select image file types, like for file picker, just stay "*" -> any  mask
            folderPicker.FileTypeFilter.Add("*");

            return await folderPicker.PickSingleFolderAsync();
        }
    }
}
