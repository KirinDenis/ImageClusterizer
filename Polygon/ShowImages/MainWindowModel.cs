using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Controls;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Windows.Storage.Pickers.Provider;

namespace ShowImages
{
    public record ImageData
    {
        public string FileName { get; init; }
    }


    public partial class MainWindowModel : ObservableObject
    {

        [ObservableProperty]
        private string hello = "Hello";

        [ObservableProperty]
        private ObservableCollection<ImageData> intCol = new();

        //private CancellationTokenSource? _cts;
        //private readonly DispatcherQueue _dispatcherQueue;
        public MainWindowModel()
        {
          //  _dispatcherQueue = DispatcherQueue.GetForCurrentThread();
            //_cts?.Cancel();
            //_cts = new CancellationTokenSource();
            _Hello();


        }

        [RelayCommand]
        public async  Task _Hello()
        {

            var imageFiles = Directory
                .EnumerateFiles(@"e:\downloads\", "*.jpg", SearchOption.AllDirectories)                
                .ToList();

            foreach (var imageFile in imageFiles)
            {
                IntCol.Add(new ImageData()
                {
                    FileName = imageFile,
                });

            }

            int i = 0;
            while (true)
            {
                Hello = "Hello + " + i.ToString();
                i++;


                IntCol.Add(new ImageData()
                {
                    FileName = @"E:\Downloads\WhatsApp Image 2026-01-14 at 2.18.22 PM.jpeg",
                });
                await Task.Delay(1000);
            }


        }

    }
}
