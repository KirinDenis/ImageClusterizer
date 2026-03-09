using ImageClusterizer.Models;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImageClusterizer
{
    /// <summary>
    /// High-performance scatter-plot canvas using DrawingVisual + VisualCollection.
    /// Handles 200k+ dots without WPF ObservableCollection/ItemsControl overhead.
    ///
    /// Architecture:
    ///   - VisualCollection child[0] = ContainerVisual with MatrixTransform (the "canvas space")
    ///     - Inside the container: one DrawingVisual per dot
    ///   - RenderTransform is NOT used on the element itself; only the container is transformed.
    ///   - OnRender() draws the static background (unaffected by canvas zoom/pan).
    ///   - Hit-testing transforms screen coords to canvas coords via inverse matrix.
    /// </summary>
    public class MapRenderCanvas : FrameworkElement
    {
        // ---- Background DP ----
        public static readonly DependencyProperty BackgroundProperty =
            DependencyProperty.Register(nameof(Background), typeof(Brush), typeof(MapRenderCanvas),
                new FrameworkPropertyMetadata(Brushes.Transparent,
                    FrameworkPropertyMetadataOptions.AffectsRender));

        public Brush Background
        {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        // ---- Items DP ----
        public static readonly DependencyProperty ItemsProperty =
            DependencyProperty.Register(nameof(Items), typeof(IReadOnlyList<MapDot>), typeof(MapRenderCanvas),
                new FrameworkPropertyMetadata(null, OnItemsChanged));

        public IReadOnlyList<MapDot>? Items
        {
            get => (IReadOnlyList<MapDot>?)GetValue(ItemsProperty);
            set => SetValue(ItemsProperty, value);
        }

        private static void OnItemsChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is MapRenderCanvas c)
                c.Render(e.NewValue as IReadOnlyList<MapDot>);
        }

        // ---- Internal state ----
        private readonly VisualCollection _hostVisuals;   // holds _container
        private readonly ContainerVisual _container;      // transformed canvas space
        private MatrixTransform _transform = new MatrixTransform();
        private Matrix _matrix = Matrix.Identity;
        private readonly List<(Rect bounds, MapDot dot)> _hitBoxes = new();

        public MapRenderCanvas()
        {
            _container = new ContainerVisual();
            _container.Transform = _transform;
            _hostVisuals = new VisualCollection(this) { _container };
            ClipToBounds = true;
        }

        // FrameworkElement needs these to expose visual children
        protected override int VisualChildrenCount => _hostVisuals.Count;
        protected override Visual GetVisualChild(int index) => _hostVisuals[index];

        // Background drawn in OnRender — not affected by _transform
        protected override void OnRender(DrawingContext dc)
        {
            dc.DrawRectangle(Background ?? Brushes.Transparent, null,
                new Rect(0, 0, ActualWidth, ActualHeight));
        }

        public void Render(IReadOnlyList<MapDot>? dots)
        {
            _container.Children.Clear();
            _hitBoxes.Clear();

            if (dots == null || dots.Count == 0) return;

            var borderPen = new Pen(new SolidColorBrush(Color.FromArgb(120, 100, 100, 100)), 0.8);
            borderPen.Freeze();

            foreach (var dot in dots)
            {
                double d = dot.Radius * 2.0;
                var center = new Point(dot.Radius, dot.Radius);

                Brush fill;
                try
                {
                    var bmp = new BitmapImage();
                    bmp.BeginInit();
                    bmp.UriSource = new Uri(dot.ThumbnailPath);
                    bmp.DecodePixelWidth = (int)Math.Max(d, 8);
                    bmp.CacheOption = BitmapCacheOption.OnLoad;
                    bmp.EndInit();
                    bmp.Freeze();
                    fill = new ImageBrush(bmp) { Stretch = Stretch.UniformToFill };
                    fill.Freeze();
                }
                catch
                {
                    fill = Brushes.SlateGray;
                }

                var visual = new DrawingVisual();
                using (var dc = visual.RenderOpen())
                {
                    var clip = new EllipseGeometry(center, dot.Radius, dot.Radius);
                    dc.PushClip(clip);
                    dc.DrawRectangle(fill, null, new Rect(0, 0, d, d));
                    dc.Pop();
                    dc.DrawEllipse(null, borderPen, center, dot.Radius, dot.Radius);
                }
                visual.Transform = new TranslateTransform(dot.X - dot.Radius, dot.Y - dot.Radius);
                _container.Children.Add(visual);
                _hitBoxes.Add((new Rect(dot.X - dot.Radius, dot.Y - dot.Radius, d, d), dot));
            }
        }

        public void SetMatrix(Matrix m)
        {
            _matrix = m;
            _transform.Matrix = m;
        }

        public Matrix GetMatrix() => _matrix;

        /// <summary>
        /// Maps a screen-space point (from mouse) to canvas space and returns the
        /// topmost dot hit, or null.
        /// </summary>
        public MapDot? HitTest(Point screenPoint)
        {
            if (!_matrix.HasInverse) return null;
            var inv = _matrix;
            inv.Invert();
            var canvas = inv.Transform(screenPoint);
            for (int i = _hitBoxes.Count - 1; i >= 0; i--)
            {
                var (_, dot) = _hitBoxes[i];
                double dx = canvas.X - dot.X;
                double dy = canvas.Y - dot.Y;
                if (dx * dx + dy * dy <= dot.Radius * dot.Radius)
                    return dot;
            }
            return null;
        }

        /// <summary>
        /// Cycles the dot to bottom or top Z-order within the canvas container.
        /// </summary>
        public void CycleZOrder(MapDot dot)
        {
            int idx = -1;
            for (int i = 0; i < _hitBoxes.Count; i++)
                if (_hitBoxes[i].dot == dot) { idx = i; break; }
            if (idx < 0) return;

            var vis = (DrawingVisual)_container.Children[idx];
            var hb = _hitBoxes[idx];
            bool wasTop = idx == _container.Children.Count - 1;

            _container.Children.RemoveAt(idx);
            _hitBoxes.RemoveAt(idx);

            if (wasTop)
            {
                _container.Children.Insert(0, vis);
                _hitBoxes.Insert(0, hb);
            }
            else
            {
                _container.Children.Add(vis);
                _hitBoxes.Add(hb);
            }
        }
    }
}
