namespace ImageClusterizer.Services;

using System;
using System.Collections.Concurrent;

/// <summary>
/// Thread-safe application log service.
/// Subscribers receive timestamped messages via the LogAdded event.
/// Consumers (ViewModel) dispatch messages to the UI thread and trim to MaxLines.
/// </summary>
public class LogService
{
    /// <summary>Maximum number of lines stored in the console panel</summary>
    public const int MaxLines = 200;

    /// <summary>
    /// Fired on any thread when a new log message is added.
    /// Subscribers must dispatch to the UI thread before updating ObservableCollection.
    /// </summary>
    public event Action<string>? LogAdded;

    private readonly object _lock = new object();

    /// <summary>
    /// Adds a timestamped log message.
    /// Format: [HH:mm:ss.fff] message
    /// Thread-safe — can be called from any thread.
    /// </summary>
    public void Log(string message)
    {
        var line = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
        System.Diagnostics.Debug.WriteLine(line);

        // Fire event outside the lock to avoid deadlocks
        Action<string>? handler;
        lock (_lock)
        {
            handler = LogAdded;
        }
        handler?.Invoke(line);
    }

    /// <summary>Signals subscribers to clear the console display</summary>
    public void Clear()
    {
        Log("--- Console cleared ---");
    }
}
