using ImageClusterizer.Models;
using ImageClusterizer.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
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
        private double _currentScale = 1.0;
        private Point _panOrigin;
        private Point _panStart;
        private bool _isPanning;

        // ---- Hover popup state ----
        private ImageVisualItem? _hoveredItem;
        private bool _dragMoved;

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
            => ApplyZoom(ZoomStep, new Point(MapViewport.ActualWidth / 2, MapViewport.ActualHeight / 2));

        private void BtnZoomOut_Click(object sender, RoutedEventArgs e)
            => ApplyZoom(1.0 / ZoomStep, new Point(MapViewport.ActualWidth / 2, MapViewport.ActualHeight / 2));

        private void BtnZoomReset_Click(object sender, RoutedEventArgs e)
        {
            _currentScale = 1.0;
            MapScale.ScaleX = 1.0;
            MapScale.ScaleY = 1.0;
            MapTranslate.X = 0;
            MapTranslate.Y = 0;
            HoverPanel.Visibility = Visibility.Collapsed;
        }

        // ========================================================================
        // MOUSE WHEEL ZOOM
        // ========================================================================

        private void MapViewport_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            var mousePos = e.GetPosition(MapViewport);
            double factor = e.Delta > 0 ? ZoomStep : 1.0 / ZoomStep;
            ApplyZoom(factor, mousePos);
            e.Handled = true;
        }

        private void ApplyZoom(double factor, Point viewportCenter)
        {
            double newScale = Math.Clamp(_currentScale * factor, ZoomMin, ZoomMax);
            if (Math.Abs(newScale - _currentScale) < 1e-9) return;

            double scaleFactor = newScale / _currentScale;
            double newTx = viewportCenter.X - scaleFactor * (viewportCenter.X - MapTranslate.X);
            double newTy = viewportCenter.Y - scaleFactor * (viewportCenter.Y - MapTranslate.Y);

            _currentScale = newScale;
            MapScale.ScaleX = newScale;
            MapScale.ScaleY = newScale;
            MapTranslate.X = newTx;
            MapTranslate.Y = newTy;
        }

        // ========================================================================
        // PAN (drag)
        // ========================================================================

        private void MapViewport_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // If click is on a dot, do not start pan
            if (e.OriginalSource is FrameworkElement fe && fe.Tag is ImageVisualItem)
                return;

            _isPanning = true;
            _dragMoved = false;
            _panOrigin = new Point(MapTranslate.X, MapTranslate.Y);
            _panStart = e.GetPosition(MapViewport);
            MapViewport.CaptureMouse();
            MapViewport.Cursor = Cursors.SizeAll;
            e.Handled = true;
        }

        private void MapViewport_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isPanning)
            {
                _isPanning = false;
                MapViewport.ReleaseMouseCapture();
                MapViewport.Cursor = Cursors.Grab;
            }
        }

        private void MapViewport_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isPanning && e.LeftButton == MouseButtonState.Pressed)
            {
                var current = e.GetPosition(MapViewport);
                var delta = current - _panStart;
                if (Math.Abs(delta.X) + Math.Abs(delta.Y) > 3)
                    _dragMoved = true;
                MapTranslate.X = _panOrigin.X + delta.X;
                MapTranslate.Y = _panOrigin.Y + delta.Y;
                return;
            }

            // Update hover panel position as mouse moves
            if (HoverPanel.Visibility == Visibility.Visible)
            {
                var pos = e.GetPosition(this);
                UpdateHoverPanelPosition(pos);
            }
        }

        private void MapViewport_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Right-click resets zoom/pan
            BtnZoomReset_Click(sender, e);
        }

        // ========================================================================
        // DOT HOVER POPUP
        // ========================================================================

        private void Dot_MouseEnter(object sender, MouseEventArgs e)
        {
            if (sender is not FrameworkElement fe || fe.Tag is not ImageVisualItem item)
                return;

            _hoveredItem = item;

            try
            {
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.UriSource = new Uri(item.ThumbnailPath);
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.DecodePixelWidth = 172;
                bmp.EndInit();
                HoverImage.Source = bmp;
            }
            catch
            {
                HoverImage.Source = null;
            }

            HoverFileName.Text = Path.GetFileName(item.FilePath);
            long kb = item.FileSize / 1024;
            HoverFileSize.Text = kb >= 1024
                ? $"{kb / 1024.0:F1} MB  |  {item.FileSize:N0} bytes"
                : $"{kb} KB  |  {item.FileSize:N0} bytes";

            UpdateHoverPanelPosition(e.GetPosition(this));
            HoverPanel.Visibility = Visibility.Visible;
        }

        private void Dot_MouseLeave(object sender, MouseEventArgs e)
        {
            _hoveredItem = null;
            HoverPanel.Visibility = Visibility.Collapsed;
        }

        private void UpdateHoverPanelPosition(Point mouseInWindow)
        {
            const double margin = 16;
            const double panelW = 188;
            const double panelH = 280;

            double x = mouseInWindow.X + margin;
            double y = mouseInWindow.Y + margin;

            x = Math.Max(margin, Math.Min(x, ActualWidth - panelW - margin));
            y = Math.Max(margin, Math.Min(y, ActualHeight - panelH - margin));

            HoverPanel.Margin = new Thickness(x, y, 0, 0);
        }

        // ========================================================================
        // DOT CLICK — open Explorer or cycle Z-order
        // ========================================================================

        private void Dot_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is not FrameworkElement fe || fe.Tag is not ImageVisualItem item)
                return;

            // Shift+Click = cycle Z-order so you can see images under this one
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                CycleDotZOrder(fe);
                e.Handled = true;
                return;
            }

            // Normal click = open Windows Explorer with file selected
            if (!_dragMoved)
                OpenInExplorer(item.FilePath);

            e.Handled = true;
        }

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

        /// <summary>
        /// Cycles the ContentPresenter Z-order within its Canvas:
        /// if already at top, move to bottom — otherwise bring to top.
        /// Lets the user reveal images hidden behind others.
        /// </summary>
        private static void CycleDotZOrder(FrameworkElement dot)
        {
            var cp = FindParentOfType<ContentPresenter>(dot);
            if (cp == null) return;

            var canvas = VisualTreeHelper.GetParent(cp) as Canvas;
            if (canvas == null) return;

            int current = Panel.GetZIndex(cp);
            int childCount = VisualTreeHelper.GetChildrenCount(canvas);
            Panel.SetZIndex(cp, current >= childCount - 1 ? 0 : childCount);
        }

        private static T? FindParentOfType<T>(DependencyObject child) where T : DependencyObject
        {
            var parent = VisualTreeHelper.GetParent(child);
            while (parent != null)
            {
                if (parent is T typed) return typed;
                parent = VisualTreeHelper.GetParent(parent);
            }
            return null;
        }
    }
}
