namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class GridViewComponent
    {
        [Parameter]
        public DetectionState DetectionState { get; set; }

        [Parameter]
        public SortBy SortBy { get; set; }

        [Parameter]
        public Models.SortOrder SortOrder { get; set; }

        [Parameter]
        public Timeframe Timeframe { get; set; }

        [Parameter]
        public DateTime? StartDateTime { get; set; }

        [Parameter]
        public DateTime? EndDateTime { get; set; }

        [Parameter]
        public string Location { get; set; } = string.Empty;

        [Parameter]
        public int MaxRecords { get; set; } = 0;

        [Parameter]
        public EventCallback OnHideReviewButton { get; set; }

        [Parameter]
        public EventCallback OnShowReviewButton { get; set; }

        private async Task OnHideReviewButtonClicked()
        {
            await OnHideReviewButton.InvokeAsync();
        }

        private async Task OnShowReviewButtonClicked()
        {
            await OnShowReviewButton.InvokeAsync();
        }

        public async Task OnReviewClicked()
        {

        }
    }
}
