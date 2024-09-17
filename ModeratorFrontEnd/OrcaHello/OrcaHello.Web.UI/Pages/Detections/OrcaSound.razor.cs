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

        protected ElementReference audioElement;


        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                IsLoading = true;

                await InvokeAsync(StateHasChanged);

                try
                {
                    ItemView = await ViewService.RetrieveDetectionAsync(Id);

                    await InvokeAsync(StateHasChanged);

                    await JSRuntime.InvokeVoidAsync("addAudioEventListenersByObject", audioElement);
                }
                catch (Exception ex)
                {
                    ReportError("Trouble loading Detection data", ex.Message);
                }

                IsLoading = false;

                await InvokeAsync(StateHasChanged);
            }
        }
    }
}
