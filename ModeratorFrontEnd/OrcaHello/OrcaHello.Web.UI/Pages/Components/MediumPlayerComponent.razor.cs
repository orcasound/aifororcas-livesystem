namespace OrcaHello.Web.UI.Pages.Components
{
    public partial class MediumPlayerComponent
    {
        [Inject]
        public IHowl Howl { get; set; } = null!;

        [Parameter]
        public DetectionItemView ItemView { get; set; } = null!;

        [Parameter]
        public string PlaybackId { get; set; } = string.Empty;

        [Parameter]
        public EventCallback<string> PlaybackIdChanged { get; set; }

        protected int soundId = -1; // the Howler sound Id of audio currently being played
        protected double PercentCompleteLine; // indicates where to draw the traveling line
        protected PlaybackState PlaybackState = PlaybackState.NotPlaying; // indicates the current playback state of the audio file
        protected string PlaybackTimer = "00:00 / 00:00"; // indicates the time within the playback
        protected double PlaybackLength = 60.00; // have not found a way to calculate the playback length before starting to play,
                                                 // so going to have to hard code it
        protected ElementReference imageRef; // reference to the rendered image

        protected override void OnInitialized()
        {
            // Register callbacks
            Howl.OnPlay += async e =>
            {
                PlaybackState = PlaybackState.Playing;
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnPause += async e =>
            {
                PlaybackState = PlaybackState.Paused;
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnStop += async e =>
            {
                PlaybackState = PlaybackState.NotPlaying;
                ResetLine();
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnEnd += async e =>
            {
                PlaybackState = PlaybackState.NotPlaying;
                ResetLine();
                await InvokeAsync(StateHasChanged);
            };
        }

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (PlaybackState == PlaybackState.Playing)
            {
                await UpdateLine();
                await InvokeAsync(StateHasChanged);
            }
        }

        protected async Task Play()
        {
            await PlaybackIdChanged.InvokeAsync(ItemView.Id);

            await JSRuntime.InvokeVoidAsync("clearAllHowls");

            HowlOptions options = new()
            {
                Sources = new string[] { ItemView.AudioUri },
                Html5 = true
            };

            soundId = await Howl.Play(options);
        }

        protected async Task Pause()
        {
            await Howl.Pause(soundId);
        }

        protected async Task Stop()
        {
            await Howl.Stop(soundId);
        }

        private async Task UpdateLine()
        {
            ValueTask<TimeSpan> currentTimeTask = Howl.GetCurrentTime(soundId);
            TimeSpan currentTime = await currentTimeTask;

            ValueTask<TimeSpan> totalTimeTask = Howl.GetTotalTime(soundId);
            TimeSpan totalTime = await totalTimeTask;

            var progress = currentTime / totalTime;

            // This is for that weird scenario when GetCurrentTime and GetTotalTime return 0;
            // in which case we don't want to update the progress line
            if (progress > 0)
            {
                PercentCompleteLine = progress * 100;
                PlaybackTimer = $"{currentTime.ToString(@"mm\:ss")} / {totalTime.ToString(@"mm\:ss")}";
            }
        }

        private void ResetLine()
        {
            PercentCompleteLine = 0;
            soundId = -1;
            PlaybackTimer = "00:00 / 00:00";
        }

        // Handle the click event on the image element
        protected async void OnImageClick(MouseEventArgs e)
        {
            if (PlaybackState == PlaybackState.Playing)
            {
                // Get the mouse position relative to the element
                var rect = await JSRuntime.InvokeAsync<DOMRect>("getBoundingClientRect", imageRef);
                var x = e.ClientX - rect.Left;
                var width = rect.Width;
                var percentage = x / width * 100;

                var position = percentage / 100 * PlaybackLength;

                await Howl.Seek(soundId, new TimeSpan(0, 0, Convert.ToInt32(position)));
            }
        }
    }
}
