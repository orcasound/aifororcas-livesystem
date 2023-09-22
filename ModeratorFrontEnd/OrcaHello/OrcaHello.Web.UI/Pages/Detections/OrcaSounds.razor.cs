namespace OrcaHello.Web.UI.Pages.Detections
{
    public partial class OrcaSounds
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; }

        [Inject]
        public AppSettings AppSettings { get; set; }

        protected IEnumerable<DetectionItemView> DetectionItemViews;
        protected int TotalDetectionCount = 0;
        protected bool IsLoading = false;

        // Detection State
        List<PicklistOption> DetectionStateOptions = new() {
            new(0, "Candidates for Review", DetectionState.Unreviewed.ToString()),
            new(1, "Confirmed Whale Calls", DetectionState.Positive.ToString()),
            new(2, "Negative Detections", DetectionState.Negative.ToString()),
            new(3, "Unknown or Unconfirmed Detections", DetectionState.Unknown.ToString()) };
        protected int? DetectionStateValue = 0; // Candidates is default

        // View Mode
        protected int? ViewMode = 0; // Tile mode is default

        // View Type
        List<PicklistOption> ViewTypeOptions = new()
        {
            new(0, "Tile View", null, "grid_view"),
            new(1, "Grid View", null, "toc")
        };
        protected int? ViewTypeValue = 0; // Tile View is default

        // Sort By
        List<PicklistOption> SortByOptions = new() { new(0, "Confidence", "confidence"), new(1, "Timestamp", "timestamp") };
        protected int? SortByValue = 1; // Timestamp is default

        // Sort Order
        List<PicklistOption> SortOrderOptions = new() { new(0, "Ascending", "ASC"), new(1, "Descending", "DESC") };
        int? SortOrderValue = 1; // Ascending is default

        // Timeframe
        List<PicklistOption> TimeframeOptions = new() { new(0, "Last 30 Minutes", "30min"), new(1, "Last 3 Hours", "3hrs"),
            new(2, "Last 6 Hours", "6hrs"), new(3, "Last 24 Hours", "24hrs"), new(4, "Last Week", "7dys"),
            new(5, "Last Month", "30dys"), new(6, "All", "all"), new(7, "Select Date Range", "range")};
        int? TimeframeValue = 2;

        // Locations
        List<PicklistOption> LocationOptions = new();
        int? LocationValue = 0;

        protected DateTime? StartDateTime;
        protected DateTime? EndDateTime;

        RadzenDataList<DetectionItemView> DetectionDataList;
        RadzenDataGrid<DetectionItemView> DetectionDataGrid;

        #region lifecycle events

        protected override void OnInitialized()
        {
            int index = 0;
            foreach (var location in AppSettings.HydrophoneLocationNames)
            {
                LocationOptions.Add(new(index++, location));
            }
            LocationOptions.Add(new(index, "All"));
            LocationValue = index;
        }

        #endregion

        #region button actions

        protected async Task OnDetectionStateChanged()
        {
            DetectionDataList.Data = null;

            if (DetectionStateValue.HasValue)
            {
                switch (DetectionStateOptions[DetectionStateValue.Value].ValueText)
                {
                    case nameof(DetectionState.Unreviewed):
                        TimeframeValue = TimeframeOptions.Where(x => x.ValueText == "6hrs").Select(x => x.Value).FirstOrDefault();
                        break;
                    case nameof(DetectionState.Positive):
                    case nameof(DetectionState.Negative):
                    case nameof(DetectionState.Unknown):
                        TimeframeValue = TimeframeOptions.Where(x => x.ValueText == "24hrs").Select(x => x.Value).FirstOrDefault();
                        break;
                }
            }

            await DetectionDataList.Reload();
        }

        protected async Task OnGridModeClicked()
        {
            ViewMode = 1;
            DetectionDataList.Data = null;
            await DetectionDataList.Reload();
        }

        protected async Task OnTileModeClicked()
        {
            ViewMode = 0;
            DetectionDataList.Data = null;
            await DetectionDataList.Reload();
        }

        protected async Task OnApplyFilterClicked()
        {
            DetectionDataList.Data = null;
            await DetectionDataList.Reload();
        }

        #endregion

        #region helpers

        private async Task LoadData(LoadDataArgs args)
        {
            IsLoading = true;

            var state = DetectionStateValue.HasValue ?
                DetectionStateOptions[DetectionStateValue.Value].ValueText : string.Empty;

            var sortBy = SortByValue.HasValue ?
                SortByOptions[SortByValue.Value].ValueText : string.Empty;

            var isDescending = SortOrderValue.HasValue ?
                SortOrderOptions[SortOrderValue.Value].ValueText == "ASC" ? false : true : true;

            var locationName = LocationValue.HasValue && LocationValue.Value < LocationOptions.Count - 1 ?
                LocationOptions[LocationValue.Value].Text : string.Empty;

            // This sets the initial value to all
            var fromDate = StartDateTime.HasValue ? StartDateTime.Value : new DateTime(1970, 1, 1);
            var toDate = EndDateTime.HasValue ? EndDateTime.Value : DateTime.UtcNow;

            if (TimeframeValue.HasValue)
            {
                switch (TimeframeOptions[TimeframeValue.Value].ValueText)
                {
                    case "30min":
                        fromDate = toDate.AddMinutes(-30);
                        break;
                    case "3hrs":
                        fromDate = toDate.AddHours(-3);
                        break;
                    case "6hrs":
                        fromDate = toDate.AddHours(-6);
                        break;
                    case "24hrs":
                        fromDate = toDate.AddHours(-24);
                        break;
                    case "7dys":
                        fromDate = toDate.AddDays(-7);
                        break;
                    case "30dys":
                        fromDate = toDate.AddDays(-30);
                        break;
                }
            }

            DetectionFilterAndPagination filterAndPagination = new()
            {
                Page = (args.Skip.Value / args.Top.Value) + 1,
                PageSize = args.Top.Value,
                State = state,
                SortBy = sortBy,
                IsDescending = isDescending,
                FromDate = fromDate,
                ToDate = toDate,
                Location = locationName
            };

            var result = await ViewService.RetrieveFilteredAndPaginatedDetectionItemViewsAsync(filterAndPagination);
            // Update the Data property
            DetectionItemViews = result.DetectionItemViews;
            // Update the count
            TotalDetectionCount = result.Count;

            IsLoading = false;
        }

        #endregion
    }
}
