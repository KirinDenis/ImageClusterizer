namespace ImageClusterizer;
using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

/// <summary>
/// Converts a double radius value to diameter (double) for Width/Height binding.
/// DotRadius -> Border Width = DotRadius * 2.
/// </summary>
[ValueConversion(typeof(double), typeof(double))]
public class DoubleToSizeConverter : IValueConverter
{
    public static readonly DoubleToSizeConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d) return d * 2.0;
        return 28.0;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

/// <summary>
/// Converts a double radius value to CornerRadius for a fully-round dot.
/// </summary>
[ValueConversion(typeof(double), typeof(CornerRadius))]
public class DoubleToCornerRadiusConverter : IValueConverter
{
    public static readonly DoubleToCornerRadiusConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d) return new CornerRadius(d);
        return new CornerRadius(14);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}

/// <summary>
/// Converts a double radius to its negative value (-radius).
/// Used for TranslateTransform to center the dot on its Canvas position:
/// Canvas.Left = X, Canvas.Top = Y, TranslateX = -radius, TranslateY = -radius.
/// </summary>
[ValueConversion(typeof(double), typeof(double))]
public class NegativeHalfConverter : IValueConverter
{
    public static readonly NegativeHalfConverter Instance = new();

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d) return -d;
        return -14.0;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotImplementedException();
}
