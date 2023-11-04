using Azure.Core;

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
    }
}
