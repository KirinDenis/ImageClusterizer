using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Dispatching;
using System.Threading;
using System.Threading.Tasks;

namespace BindingTest
{
    public partial class MainWindowModel : ObservableObject
    {
        [ObservableProperty]
        private string somethingRun = "run...";

        [ObservableProperty]
        private string somethingRun2 = "run...";


        private CancellationTokenSource? _cts;
        private readonly DispatcherQueue _dispatcherQueue;
        public MainWindowModel()
        {
            _dispatcherQueue = DispatcherQueue.GetForCurrentThread();

            _cts?.Cancel();
            _cts = new CancellationTokenSource();

            _ = DoSome();
            _ = DoSome2();
        }

        [RelayCommand]
        public async Task DoSome()
        {
            int i = 0;
            while (!_cts.Token.IsCancellationRequested)
            {
                SomethingRun = i.ToString();
                i++;
                await Task.Delay(1, _cts.Token);
            }
        }

        [RelayCommand]
        public async Task DoSome2()
        {
            int i = 0;
            while (!_cts.Token.IsCancellationRequested)
            {
                SomethingRun2 = i.ToString();
                i++;
                await Task.Delay(1, _cts.Token);
            }
        }


    }
}
