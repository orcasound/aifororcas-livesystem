namespace OrcaHello.Web.UI.Pages.Detections
{
    [ExcludeFromCodeCoverage]
    public partial class OrcaSound
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Parameter]
        public string Id { get; set; } = string.Empty; // The ID of the detection being displayed

        protected DetectionItemView ItemView = null!; // The corresponding detection object

        protected bool IsLoading = false; // flag indicating whether the detection object is being loaded

        protected string LinkUrl { get => $"{NavManager.BaseUri}orca_sounds/{Id}"; } // calculated address of the record link

        private PlaybackState PlaybackState = PlaybackState.NotPlaying; // indicates the current playback state of the audio file

        private async Task Play()
        {
            PlaybackState = PlaybackState.Playing;
            await JSRuntime.InvokeVoidAsync("playAudio", ItemView.Id);
        }

        private async Task Pause()
        {
            PlaybackState = PlaybackState.Paused;
            await JSRuntime.InvokeVoidAsync("pauseAudio", ItemView.Id);
        }

        private async Task Stop()
        {
            PlaybackState = PlaybackState.NotPlaying;
            await JSRuntime.InvokeVoidAsync("stopAudio", ItemView.Id);
        }

        protected override async Task OnInitializedAsync()
        {
            IsLoading = true;

            try
            {
                ItemView = await ViewService
                    .RetrieveDetectionAsync(Id);
            }
            catch (Exception ex)
            {
                ReportError("Trouble loading Detection data", ex.Message);
            }

            IsLoading = false;
        }
    }
}
