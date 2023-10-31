namespace OrcaHello.Web.UI.Pages.Dashboard.Components
{
    public partial class DetectionMetricsComponent
    { 
        [Inject]
        public IMetricsViewService ViewService { get; set; } = null!;

        [Parameter]
        public MetricsStateView StateView { get; set; } = null!;

        #region lifecycle events

        protected override async Task OnParametersSetAsync()
        {
            await LoadMetricsAsync();
        }

        #endregion

        #region data loaders

        protected async Task LoadMetricsAsync()
        {
            StateView.IsLoading = true;

            MetricsByDateRequest request = new()
            {
                FromDate = StateView.FromDate,
                ToDate = StateView.ToDate
            };

            try
            {
                MetricsItemViewResponse response = await ViewService.RetrieveFilteredMetricsAsync(request);

                StateView.MetricsItemViews = response.MetricsItemViews;
                StateView.FillColors = response.MetricsItemViews.Select(x => x.Color).ToList();
            }
            catch(Exception ex) 
            {
                ReportError("Trouble loading Metrics data", ex.Message);
            }

            StateView.IsLoading = false;
        }

        #endregion
    }
}
