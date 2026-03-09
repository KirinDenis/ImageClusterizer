using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImageClusterizer
{
    public partial class MainWindow : Window
    {
        public MainViewModel ViewModel { get; private set; }

        // ---- Zoom / Pan state ----
        private const double ZoomStep = 1.25;
        private const double ZoomMin = 0.05;
        private const double ZoomMax = 50.0;
        private Matrix _matrix = Matrix.Identity;
        private Point _panOrigin;
        private Point _panStart;
        private bool _isPanning;
        private bool _dragMoved;

        // ---- Hover state ----
        private MapDot? _hoveredDot;

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = (App.Services?.GetRequiredService<MainViewModel>())!;
            DataContext = this;
            ViewModel.ConsoleLines.CollectionChanged += OnConsoleLinesChanged;
        }

        private void OnConsoleLinesChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            ConsoleScrollViewer?.ScrollToBottom();
        }

        // ========================================================================
        // ZOOM BUTTONS
        // ========================================================================
        private void BtnZoomIn_Click(object sender, RoutedEventArgs e)
            => ApplyZoom(ZoomStep, new Point(MapCanvas.ActualWidth / 2, MapCanvas.ActualHeight / 2));

        private void BtnZoomOut_Click(object sender, RoutedEventArgs e)
            => ApplyZoom(1.0 / ZoomStep, new Point(MapCanvas.ActualWidth / 2, MapCanvas.ActualHeight / 2));

        private void BtnZoomReset_Click(object sender, RoutedEventArgs e)
        {
            _matrix = Matrix.Identity;
            MapCanvas.SetMatrix(_matrix);
            HoverPanel.Visibility = Visibility.Collapsed;
            _hoveredDot = null;
            ViewModel.ZoomText = "100%";
        }

        // ========================================================================
        // MOUSE WHEEL ZOOM
        // ========================================================================
        private void MapViewport_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            var mousePos = e.GetPosition(MapCanvas);
            double factor = e.Delta > 0 ? ZoomStep : 1.0 / ZoomStep;
            ApplyZoom(factor, mousePos);
            e.Handled = true;
        }

        private void ApplyZoom(double factor, Point center)
        {
            double currentScale = _matrix.M11;
            double newScale = Math.Clamp(currentScale * factor, ZoomMin, ZoomMax);
            if (Math.Abs(newScale - currentScale) < 1e-9) return;

            double scaleFactor = newScale / currentScale;
            _matrix.ScaleAtPrepend(scaleFactor, scaleFactor, center.X, center.Y);
            MapCanvas.SetMatrix(_matrix);
            ViewModel.ZoomText = $"{newScale * 100:F0}%";
        }

        // ========================================================================
        // PAN (drag)
        // ========================================================================
        private void MapViewport_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _isPanning = true;
            _dragMoved = false;
            _panOrigin = new Point(_matrix.OffsetX, _matrix.OffsetY);
            _panStart = e.GetPosition(MapCanvas);
            MapCanvas.CaptureMouse();
            MapCanvas.Cursor = Cursors.SizeAll;
            e.Handled = true;
        }

        private void MapViewport_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (!_isPanning) return;
            _isPanning = false;
            MapCanvas.ReleaseMouseCapture();
            MapCanvas.Cursor = Cursors.Hand;

            if (!_dragMoved)
            {
                // Treat as click
                var pos = e.GetPosition(MapCanvas);
                var dot = MapCanvas.HitTest(pos);
                if (dot != null)
                {
                    if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
                        MapCanvas.CycleZOrder(dot);
                    else
                        OpenInExplorer(dot.FilePath);
                }
            }
        }

        private void MapViewport_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isPanning && e.LeftButton == MouseButtonState.Pressed)
            {
                var current = e.GetPosition(MapCanvas);
                var delta = current - _panStart;
                if (Math.Abs(delta.X) + Math.Abs(delta.Y) > 3)
                    _dragMoved = true;

                var m = _matrix;
                m.OffsetX = _panOrigin.X + delta.X;
                m.OffsetY = _panOrigin.Y + delta.Y;
                _matrix = m;
                MapCanvas.SetMatrix(_matrix);
                return;
            }

            // Hit test for hover panel
            var mousePos = e.GetPosition(MapCanvas);
            var hitDot = MapCanvas.HitTest(mousePos);

            if (hitDot != _hoveredDot)
            {
                _hoveredDot = hitDot;
                if (hitDot != null)
                    ShowHoverPanel(hitDot, e.GetPosition(this));
                else
                    HoverPanel.Visibility = Visibility.Collapsed;
            }
            else if (hitDot != null && HoverPanel.Visibility == Visibility.Visible)
            {
                UpdateHoverPanelPosition(e.GetPosition(this));
            }
        }

        private void MapViewport_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            BtnZoomReset_Click(sender, e);
        }

        // ========================================================================
        // HOVER PANEL
        // ========================================================================
        private void ShowHoverPanel(MapDot dot, Point mouseInWindow)
        {
            try
            {
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.UriSource = new Uri(dot.ThumbnailPath);
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.DecodePixelWidth = 176;
                bmp.EndInit();
                HoverImage.Source = bmp;
            }
            catch
            {
                HoverImage.Source = null;
            }

            HoverFileName.Text = Path.GetFileName(dot.FilePath);
            long kb = dot.FileSize / 1024;
            HoverFileSize.Text = kb >= 1024
                ? $"{kb / 1024.0:F1} MB | {dot.FileSize:N0} bytes"
                : $"{kb} KB | {dot.FileSize:N0} bytes";

            UpdateHoverPanelPosition(mouseInWindow);
            HoverPanel.Visibility = Visibility.Visible;
        }

        private void UpdateHoverPanelPosition(Point mouseInWindow)
        {
            const double margin = 16;
            const double panelW = 200;
            const double panelH = 290;

            double x = mouseInWindow.X + margin;
            double y = mouseInWindow.Y + margin;
            x = Math.Max(margin, Math.Min(x, ActualWidth - panelW - margin));
            y = Math.Max(margin, Math.Min(y, ActualHeight - panelH - margin));
            HoverPanel.Margin = new Thickness(x, y, 0, 0);
        }

        // ========================================================================
        // HELPERS
        // ========================================================================
        private static void OpenInExplorer(string filePath)
        {
            try
            {
                if (File.Exists(filePath))
                    Process.Start("explorer.exe", $"/select,\"{filePath}\"");
                else if (Directory.Exists(Path.GetDirectoryName(filePath)!))
                    Process.Start("explorer.exe", Path.GetDirectoryName(filePath)!);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"OpenInExplorer: {ex.Message}");
            }
        }
    }
}
