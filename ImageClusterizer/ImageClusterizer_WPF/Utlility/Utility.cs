using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
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

        public static string? SelectFolderDiagoAsync()
        {
            var dialog = new OpenFolderDialog()
            {
                Title = "Select folder with images",
                Multiselect = false
            };

            return dialog.ShowDialog() == true
                ? dialog.FolderName
                : null;
        }
        public static bool IsImageFile(string filePath)
        {
            string extension = Path.GetExtension(filePath).ToLowerInvariant();
            return extension is ".jpg" or ".jpeg" or ".gif" or ".png" or ".bmp";
        }

    }
}
