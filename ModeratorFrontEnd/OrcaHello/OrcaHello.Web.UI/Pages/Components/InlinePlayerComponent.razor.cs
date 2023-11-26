namespace OrcaHello.Web.UI.Pages.Components
{
    [ExcludeFromCodeCoverage]
    public partial class InlinePlayerComponent
    {
        [Inject]
        public IHowl Howl { get; set; } = null!;

        [Parameter]
        public string AudioUri { get; set; } = string.Empty;

        [Parameter]
        public string SpectrogramUri { get; set; } = string.Empty;

        [Parameter]
        public string Id { get; set; } = string.Empty;  

        [Parameter]
        public string PlaybackId { get; set; } = string.Empty;

        [Parameter]
        public EventCallback<string> PlaybackIdChanged { get; set; }

        protected int soundId = -1; // the Howler sound Id of audio currently being played

        protected PlaybackState PlaybackState = PlaybackState.NotPlaying; // indicates the current playback state of the audio file

        protected ElementReference imageRef; // reference to the rendered image

        protected override void OnInitialized()
        {
            // Register callbacks
            Howl.OnPlay += async e =>
            {
                PlaybackState = PlaybackState.Playing;
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnStop += async e =>
            {
                PlaybackState = PlaybackState.NotPlaying;
                await InvokeAsync(StateHasChanged);
            };

            Howl.OnEnd += async e =>
            {
                PlaybackState = PlaybackState.NotPlaying;
                await InvokeAsync(StateHasChanged);
            };
        }

        protected async Task Play()
        {
            await PlaybackIdChanged.InvokeAsync(Id);

            await JSRuntime.InvokeVoidAsync("clearAllHowls");

            HowlOptions options = new()
            {
                Sources = new string[] { AudioUri },
                Html5 = true
            };

            soundId = await Howl.Play(options);
        }

        protected async Task Stop()
        {
            await Howl.Stop(soundId);
        }
    }
}
