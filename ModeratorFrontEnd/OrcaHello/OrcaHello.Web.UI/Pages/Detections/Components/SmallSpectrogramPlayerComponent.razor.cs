namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class SmallSpectrogramPlayerComponent
    {
        [Parameter]
        public DetectionItemView DetectionItemView { get; set; } = null!;

        private string _id;

        // These are needed to makes sure the components have a unique ID for the [age
        private string SpectrogramPlayerId => $"spectrogram-player-{_id}";
        private string SpectrogramImageId => $"spectrogram-image-{_id}";
        private string SpectrogramProgressIndicatorId => $"spectrogram-progress-indicator-{_id}";
        private string SpectrogramProgressTimerId => $"spectrogram-progress-timer-{_id}";
        private string SpectrogramControlsId => $"spectrogram-controls-{_id}";
        private string SpectrogramPlayButtonId => $"spectrogram-play-button-{_id}";
        private string SpectrogramPauseButtonId => $"spectrogram-pause-button-{_id}";
        private string SpectrogramStopButtonId => $"spectrogram-stop-button-{_id}";
        private string SpectrogramVolumeControlId => $"spectrogram-volume-control-{_id}";

        protected override void OnParametersSet()
        {
            _id = DetectionItemView.Id;
        }

        private async Task PlaySpectrogram()
        {
            await JSRuntime.InvokeVoidAsync("StartSmallSpectrogramPlayback", _id, DetectionItemView.AudioUri);
        }

        private async Task PauseSpectrogram()
        {
            await JSRuntime.InvokeVoidAsync("PauseSmallSpectrogramPlayback");
        }

        private async Task StopSpectrogram()
        {
            await JSRuntime.InvokeVoidAsync("StopSmallSpectrogramPlayback");
        }
    }
}
