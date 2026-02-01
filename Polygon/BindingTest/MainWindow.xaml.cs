using Microsoft.UI.Xaml;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace BindingTest
{
    public sealed partial class MainWindow : Window
    {
        public MainWindowModel ViewModel { get; }

        public MainWindow()
        {
            InitializeComponent();

            ViewModel = new MainWindowModel();
        }
    }
}
