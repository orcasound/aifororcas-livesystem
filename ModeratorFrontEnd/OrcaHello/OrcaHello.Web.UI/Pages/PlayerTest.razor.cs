namespace OrcaHello.Web.UI.Pages
{
    public partial class PlayerTest
    {
        [Inject]
        IJSRuntime JSRuntime { get; set; }

        private string audioFile = "https://livemlaudiospecstorage.blob.core.windows.net/audiowavs/rpi_port_townsend_2020_11_22_12_22_38_PST.wav"; // Replace with your actual .wav file path

        private async Task PlayAudio()
        {
            await JSRuntime.InvokeVoidAsync("howlerInterop.playSound", audioFile);
            await JSRuntime.InvokeVoidAsync("howlerInterop.startProgress");
        }

        private async Task PauseAudio()
        {
            await JSRuntime.InvokeVoidAsync("howlerInterop.pauseSound");
        }

        private async Task StopAudio()
        {
            await JSRuntime.InvokeVoidAsync("howlerInterop.stopSound");
        }


        //private bool isPlaying = false;
        //private DotNetObjectReference<PlayerTest> objRef;

        //protected override void OnInitialized()
        //{
        //    objRef = DotNetObjectReference.Create(this);
        //}

        //private async Task PlayAudio()
        //{
        //    await JSRuntime.InvokeVoidAsync("howlerInterop.playSound", audioFile);
        //    isPlaying = true;
        //}

        //private async Task PauseAudio()
        //{
        //    if (isPlaying)
        //    {
        //        await JSRuntime.InvokeVoidAsync("howlerInterop.pauseSound");
        //        isPlaying = false;
        //    }
        //}

        //private async Task StopAudio()
        //{
        //    await JSRuntime.InvokeVoidAsync("howlerInterop.stopSound");
        //    isPlaying = false;
        //}

        //public async ValueTask DisposeAsync()
        //{
        //    objRef?.Dispose();
        //}
    }
}
