namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class GridViewComponent
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Parameter]
        public Filters Filters { get; set; } = null!;

        [Parameter]
        public EventCallback OnShowReviewButton { get; set; }

        protected bool IsLoading = false; // Flag for spinner
        protected IEnumerable<DetectionItemView> DetectionItemViews = null!; // List of items
        RadzenDataGrid<DetectionItemView> DetectionDataGrid = null!; // Reference to the component
        IList<DetectionItemView> SelectedDetectionItemViews = new List<DetectionItemView>(); // Which items have been selected in the grid
        protected int TotalDetectionCount = 0; // The total number of records for the applied filter
        protected Filters _currentFilters = null!; // So that only a change in Filters triggers a reload

        #region lifecycle events

        protected override async Task OnInitializedAsync()
        {
            // TODO: Get role membership

            await OnShowReviewButton.InvokeAsync();
        }

        protected override async Task OnParametersSetAsync()
        {
            if(_currentFilters != Filters)
            {
                _currentFilters = Filters;

                if (DetectionDataGrid?.Data != null)
                    await DetectionDataGrid.Reload();
            }
        }

        #endregion

        protected void OnAllItemsChanged(bool? value)
        {
            SelectedDetectionItemViews = value == true ? DetectionItemViews.ToList() : null!;
        }

        public async Task OnReviewClicked()
        {
            if(!SelectedDetectionItemViews.Any())
            {
                NotificationService.Notify(
                    new NotificationMessage
                    {
                        Style = "position:absolute; left: vh:50%; vw: 50%;",
                        Severity = NotificationSeverity.Error,
                        Summary = "Error Summary",
                        Detail = "Error Detail",
                        Duration = 40000,

                    }
               );
            }
        }

        //async Task ReloadData()
        //{
        //    DetectionDataGrid.Data = null;
        //    await DetectionDataGrid.Reload();
        //}

        private async Task LoadData(LoadDataArgs args)
        {
            IsLoading = true;
            await InvokeAsync(StateHasChanged);

            // This sets the initial value to Range (if populated) or All (if not)
            var fromDate = Filters.StartDateTime.HasValue ? Filters.StartDateTime.Value : new DateTime(1970, 1, 1);
            var toDate = Filters.EndDateTime.HasValue ? Filters.EndDateTime.Value : DateTime.UtcNow;

            if(Filters.Timeframe != Timeframe.All && Filters.Timeframe != Timeframe.Range)
                fromDate = toDate.Adjust(Filters.Timeframe);

            DetectionFilterAndPagination filterAndPagination = new()
            {
                Page = (args.Skip.Value / args.Top.Value) + 1,
                PageSize = Filters.MaxRecords,
                State = Filters.DetectionState.ToString(),
                SortBy = Filters.SortBy.ToString().ToLower(),
                IsDescending = Filters.SortOrder == Models.SortOrder.Desc,
                FromDate = fromDate,
                ToDate = toDate,
                Location = Filters.Location == "All" ? string.Empty : Filters.Location
            };

            var result = await ViewService.
                RetrieveFilteredAndPaginatedDetectionItemViewsAsync(filterAndPagination);
            
            // Update the Data property
            DetectionItemViews = result.DetectionItemViews;

            // Update the count
            TotalDetectionCount = result.Count;

            IsLoading = false;
            await InvokeAsync(StateHasChanged);
        }
    }
}
