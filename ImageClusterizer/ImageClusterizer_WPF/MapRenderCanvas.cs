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
    /// Each dot is a pre-rendered DrawingVisual. Zoom/Pan via MatrixTransform.
    /// Hit-testing is O(n) linear scan from top to bottom Z-order.
    /// </summary>
    public class MapRenderCanvas : FrameworkElement
    {
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

        private readonly VisualCollection _visuals;
        private MatrixTransform _transform = new MatrixTransform();
        private Matrix _matrix = Matrix.Identity;
        private readonly List<(Rect bounds, MapDot dot)> _hitBoxes = new();

        public MapRenderCanvas()
        {
            _visuals = new VisualCollection(this);
            RenderTransform = _transform;
            ClipToBounds = true;
        }

        protected override int VisualChildrenCount => _visuals.Count;
        protected override Visual GetVisualChild(int index) => _visuals[index];

        public void Render(IReadOnlyList<MapDot>? dots)
        {
            _visuals.Clear();
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
                _visuals.Add(visual);
                _hitBoxes.Add((new Rect(dot.X - dot.Radius, dot.Y - dot.Radius, d, d), dot));
            }
        }

        public void SetMatrix(Matrix m)
        {
            _matrix = m;
            _transform.Matrix = m;
        }

        public Matrix GetMatrix() => _matrix;

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

        public void CycleZOrder(MapDot dot)
        {
            int idx = -1;
            for (int i = 0; i < _hitBoxes.Count; i++)
                if (_hitBoxes[i].dot == dot) { idx = i; break; }
            if (idx < 0) return;

            var vis = _visuals[idx];
            var hb = _hitBoxes[idx];
            bool wasTop = idx == _visuals.Count - 1;
            _visuals.RemoveAt(idx);
            _hitBoxes.RemoveAt(idx);
            if (wasTop)
            {
                _visuals.Insert(0, vis);
                _hitBoxes.Insert(0, hb);
            }
            else
            {
                _visuals.Add(vis);
                _hitBoxes.Add(hb);
            }
        }
    }
}
