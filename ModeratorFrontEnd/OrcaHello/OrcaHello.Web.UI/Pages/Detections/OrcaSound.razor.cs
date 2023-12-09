namespace OrcaHello.Web.UI.Pages.Detections
{
    [ExcludeFromCodeCoverage]
    public partial class OrcaSound
    {
        [Inject]
        public IHowl Howl { get; set; } = null!;

        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Parameter]
        public string Id { get; set; } = string.Empty; // The ID of the detection being displayed

        protected DetectionItemView ItemView = null!; // The corresponding detection object

        protected bool IsLoading = false; // flag indicating whether the detection object is being loaded

        protected string LinkUrl { get => $"{NavManager.BaseUri}orca_sounds/{Id}"; } // calculated address of the record link

        protected int soundId = -1; // the Howler sound Id of audio currently being played
        private double PercentCompleteLine; // indicates where to draw the traveling line
        private PlaybackState PlaybackState = PlaybackState.NotPlaying; // indicates the current playback state of the audio file
        private string PlaybackTimer = "00:00 / 00:00"; // indicates the time within the playback
        private readonly double PlaybackLength = 60.00; // have not found a way to calculate the playback length before starting to play,
                                                        // so going to have to hard code it
        private TimeSpan CurrentTime = TimeSpan.Zero;
        private ElementReference imageRef; // reference to the rendered image

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
                await ResetLine();
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnEnd += async e =>
            {
                PlaybackState = PlaybackState.NotPlaying;
                await ResetLine();
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

        protected async Task Test()
        {
            await JSRuntime.InvokeVoidAsync("repositionHowl", soundId, (new TimeSpan(0, 0, 20)).TotalMilliseconds);
            //await Howl.Seek(soundId, new TimeSpan(0, 0, 20));
        }

        protected async Task Play()
        {
            await ResetLine();

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
            var currentTime = await currentTimeTask;
            if (currentTime != TimeSpan.Zero)
                CurrentTime = currentTime;

            ValueTask<TimeSpan> totalTimeTask = Howl.GetTotalTime(soundId);
            TimeSpan totalTime = await totalTimeTask;

            var progress = CurrentTime / totalTime;

            // This is for that weird scenario when GetCurrentTime and GetTotalTime return 0;
            // in which case we don't want to update the progress line
            if (progress > 0)
            {
                PercentCompleteLine = progress * 100;
                PlaybackTimer = $"{CurrentTime:mm\\:ss} / {totalTime:mm\\:ss}";
            }
        }

        private async Task ResetLine()
        {
            PercentCompleteLine = 0;
            soundId = -1;
            PlaybackTimer = "00:00 / 00:00";
            await JSRuntime.InvokeVoidAsync("clearAllHowls");
        }

        // Handle the click event on the image element
        private async void OnImageClick(MouseEventArgs e)
        {
            if (PlaybackState == PlaybackState.Playing)
            {
                // Get the mouse position relative to the element
                var rect = await JSRuntime.InvokeAsync<DOMRect>("getBoundingClientRect", imageRef);
                var x = e.ClientX - rect.Left;
                var width = rect.Width;

                var position = Convert.ToInt32(x / width * PlaybackLength) * 1000;
                var timespan = TimeSpan.FromMilliseconds(position);

                await Howl.Seek(soundId, timespan);
            }
        }
    }
}
