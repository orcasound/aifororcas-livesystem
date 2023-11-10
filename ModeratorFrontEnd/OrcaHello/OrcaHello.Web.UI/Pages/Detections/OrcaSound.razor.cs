namespace OrcaHello.Web.UI.Pages.Detections
{
    public partial class OrcaSound
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Parameter]
        public string Id { get; set; } = string.Empty;

        protected DetectionItemView ItemView = null!;

        protected bool IsLoading = false;

        protected string LinkUrl { get => $"{NavManager.BaseUri}orca_sounds/{Id}"; }

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

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if(firstRender)
            {
                await JSRuntime.InvokeVoidAsync("SetupPlayback", ItemView.AudioUri, ItemView.Id);
            }
        }

        public async ValueTask DisposeAsync()
        {
            await JSRuntime.InvokeVoidAsync("StopGridAudioPlayback");
        }
    }
}
