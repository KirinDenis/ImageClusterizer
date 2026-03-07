using Microsoft.Win32;
using System.IO;

namespace ImageClusterizer.Utlility
{
    public static class Utility
    {
        /// <summary>
        /// Opens a folder picker dialog and returns the selected folder path, or null if cancelled.
        /// </summary>
        public static string? SelectFolderDiagoAsync()
        {
            var dialog = new OpenFolderDialog()
            {
                Title      = "Select folder with images",
                Multiselect = false
            };

            return dialog.ShowDialog() == true ? dialog.FolderName : null;
        }

        /// <summary>
        /// Returns true if the file has a supported image extension
        /// </summary>
        public static bool IsImageFile(string filePath)
        {
            string extension = Path.GetExtension(filePath).ToLowerInvariant();
            return extension is ".jpg" or ".jpeg" or ".gif" or ".png" or ".bmp";
        }
    }
}
