namespace OrcaHello.Web.UI.Pages.Dashboard
{
    [ExcludeFromCodeCoverage]
    public partial class Hydrophones
    {
        [Inject]
        public IHydrophoneViewService ViewService { get; set; }

        protected List<HydrophoneItemView> HydrophoneItemViews;
        protected bool IsLoading = false;

        RadzenDataGrid<HydrophoneItemView> grid;

        protected int Zoom = 3;

        #region lifecycle events

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                await LoadDataAsync();
            }
        }

        #endregion

        #region helpers

        private async Task LoadDataAsync()
        {
            try
            {
                // blank existing data
                IsLoading = true;
                HydrophoneItemViews = null!;
                await InvokeAsync(StateHasChanged);

                // load data
                HydrophoneItemViews = await ViewService.RetrieveAllHydrophoneViewsAsync();

                // finish the process
                IsLoading = false;
                await InvokeAsync(StateHasChanged);
            }
            catch (Exception exception)
            {
                // Make the page show 'no records found' on any load exception
                HydrophoneItemViews = new();
                IsLoading = false;
                LogAndReportUnknownException(exception);
            }
        }

        #endregion
    }
}
