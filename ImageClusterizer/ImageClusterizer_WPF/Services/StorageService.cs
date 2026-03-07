namespace ImageClusterizer.Services;

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// Owns all file system path resolution for the application.
/// All persistent data lives under a structured subfolder next to the executable.
///
/// Layout:
///   <AppBaseDirectory>/
///     data/
///       vectors.db          - LiteDB vector database
///     thumbnails/
///       <sha256_of_path>.jpg - 224x224 JPEG thumbnail cache
/// </summary>
public class StorageService
{
    private readonly string _baseDirectory;

    public StorageService()
    {
        _baseDirectory = AppContext.BaseDirectory;
        EnsureDirectoriesExist();
    }

    // ---- Public paths ----

    /// <summary>Full path to the LiteDB database file</summary>
    public string DatabasePath =>
        Path.Combine(_baseDirectory, "data", "vectors.db");

    /// <summary>Full path to the thumbnails cache folder</summary>
    public string ThumbnailsFolder =>
        Path.Combine(_baseDirectory, "thumbnails");

    // ---- Thumbnail path resolution ----

    /// <summary>
    /// Returns the full path where the thumbnail for the given original image should be stored.
    /// Uses SHA256 of the original file path as filename to avoid separator issues.
    /// Example: "C:\Photos\cat.jpg" -> "{ThumbnailsFolder}\a3f9b2....jpg"
    /// </summary>
    public string GetThumbnailPath(string imageFilePath)
    {
        var hash = ComputePathHash(imageFilePath);
        return Path.Combine(ThumbnailsFolder, hash + ".jpg");
    }

    /// <summary>Returns true if a thumbnail already exists for this image path</summary>
    public bool ThumbnailExists(string imageFilePath)
        => File.Exists(GetThumbnailPath(imageFilePath));

    // ---- Data cleanup ----

    /// <summary>
    /// Deletes all stored data: the vectors database and all thumbnail files.
    /// Original user image files are NOT touched.
    /// </summary>
    public async Task ClearAllDataAsync()
    {
        await Task.Run(() =>
        {
            // Delete LiteDB database
            if (File.Exists(DatabasePath))
            {
                File.Delete(DatabasePath);
            }

            // Delete all thumbnail files
            if (Directory.Exists(ThumbnailsFolder))
            {
                foreach (var file in Directory.GetFiles(ThumbnailsFolder, "*.jpg"))
                {
                    File.Delete(file);
                }
            }
        });
    }

    // ---- Storage info ----

    /// <summary>Returns the size of the database file as a human-readable string (KB / MB)</summary>
    public string GetDatabaseSizeText()
    {
        if (!File.Exists(DatabasePath))
            return "0 KB";

        var bytes = new FileInfo(DatabasePath).Length;

        if (bytes >= 1_048_576)
            return $"{bytes / 1_048_576.0:F1} MB";

        return $"{bytes / 1024.0:F0} KB";
    }

    // ---- Private helpers ----

    private void EnsureDirectoriesExist()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(DatabasePath)!);
        Directory.CreateDirectory(ThumbnailsFolder);
    }

    /// <summary>Computes a short SHA256 hex string from the file path string (not file content)</summary>
    private static string ComputePathHash(string filePath)
    {
        var bytes = Encoding.UTF8.GetBytes(filePath.ToLowerInvariant());
        var hash  = SHA256.HashData(bytes);
        // Use first 16 bytes (32 hex chars) — collision-free for practical use
        return Convert.ToHexString(hash)[..32].ToLowerInvariant();
    }
}
