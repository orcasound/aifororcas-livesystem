namespace OrcaHello.Web.UI.Pages.Detections
{
    public partial class OrcaSounds
    {
        [Inject]
        public AppSettings AppSettings { get; set; } = null!;

        // View Mode
        protected ViewMode SelectedViewMode = ViewMode.GridView;

        // Detection State
        protected List<DropdownOption> DetectionStateDropdownOptions = new() {
            new(DetectionState.Unreviewed, "Unreviewed Calls"),
            new(DetectionState.Positive, "Verfied Calls"),
            new(DetectionState.Negative, "Rejected Calls"),
            new(DetectionState.Unknown, "Unclear Calls" )};

        protected DetectionState SelectedDetectionState = DetectionState.Unreviewed;

        // Sort By
        protected List<DropdownOption> SortByDropdownOptions = new()
        {
            new(SortBy.Confidence, "Confidence"),
            new(SortBy.Timestamp, "Timestamp")
        };

        protected SortBy SelectedSortBy = SortBy.Timestamp;
        protected SortBy PassedSortBy;

        // Sort Order
        protected List<DropdownOption> SortOrderDropdownOptions = new() 
        { 
            new(Models.SortOrder.Asc, "Ascending"), 
            new(Models.SortOrder.Desc, "Descending") 
        };

        protected Models.SortOrder SelectedSortOrder = Models.SortOrder.Asc;
        protected Models.SortOrder PassedSortOrder;

        // Timeframe
        protected List<DropdownOption> TimeframeDropdownOptions = new()
        {
            new(Timeframe.ThirtyMinutes, "Last 30 Minutes"),
            new(Timeframe.ThreeHours, "Last 3 Hours"),
            new(Timeframe.SixHours, "Last 6 hours"),
            new(Timeframe.TwentyFourHours, "Last 24 hours"),
            new(Timeframe.SevenDays, "Last Week"),
            new(Timeframe.ThirtyDays, "Last Month"),
            new(Timeframe.All, "All"),
            new(Timeframe.Range, "Select Date Range")
        };

        protected Timeframe SelectedTimeframe = Timeframe.SixHours;
        protected Timeframe PassedTimeframe;

        // Datetime Range

        protected DateTime? SelectedStartDateTime = null;
        protected DateTime? SelectedEndDateTime = null;
        protected DateTime? PassedStartDateTime = null;
        protected DateTime? PassedEndDateTime = null;

        // Locations
        List<DropdownOption> LocationDropdownOptions = new();
        protected int SelectedLocation;
        protected string PassedLocation = string.Empty;

        // Maximum Records
        List<DropdownOption> MaxRecordsDropdownOptions = new() 
        { 
            new(10, "10"),
            new(25, "25"),
            new(50, "50"),
            new(100, "100"),
            new(250, "250"),
            new(500, "500"),
            new(1000, "1000")
        };

        protected int SelectedMaxRecords = 10;
        protected int PassedMaxRecords;

        // Review button
        protected bool IsReviewButtonVisible = false;

        // Reference to GridViewComponent
        protected GridViewComponent GridView = null!;

        #region lifecycle events

        protected override void OnInitialized()
        {
            int index = 0;
            foreach (var location in AppSettings.HydrophoneLocationNames)
            {
                LocationDropdownOptions.Add(new(index++, location));
            }
            LocationDropdownOptions.Add(new(index, "All"));
            SelectedLocation = index;

            // Initially sync the passed values with the selected values
            OnApplyFilterClicked();

        }

        #endregion

        #region button actions

        // Needed in order to select the correct default timeframe

        protected void OnDetectionStateChanged()
        {
            SelectedStartDateTime = null;
            SelectedEndDateTime = null;

            switch (SelectedDetectionState)
            {
                case DetectionState.Unreviewed:
                    SelectedTimeframe = Timeframe.SixHours;
                    break;
                case DetectionState.Positive:
                case DetectionState.Negative:
                case DetectionState.Unknown:
                    SelectedTimeframe = Timeframe.TwentyFourHours;
                    break;
            }

            OnApplyFilterClicked();
        }

        protected void OnTimeframeChanged()
        {
            if(SelectedTimeframe != Timeframe.Range)
            {
                SelectedStartDateTime = null;
                SelectedEndDateTime = null;
            } 
            else if(SelectedStartDateTime == null || SelectedEndDateTime == null)
            {
                SelectedStartDateTime = DateTime.UtcNow.AddDays(-1);
                SelectedEndDateTime = DateTime.UtcNow;
            }
        }

        protected void OnApplyFilterClicked()
        {
            PassedSortBy = SelectedSortBy;
            PassedSortOrder = SelectedSortOrder;
            PassedTimeframe = SelectedTimeframe;
            PassedStartDateTime = SelectedStartDateTime;
            PassedEndDateTime = SelectedEndDateTime;
            PassedLocation = LocationDropdownOptions[SelectedLocation].Text;
            PassedMaxRecords = SelectedMaxRecords;
        }

        protected void OnGridViewClicked()
        {
            SelectedViewMode = ViewMode.GridView;
        }

        protected void OnTileViewClicked()
        {
            IsReviewButtonVisible = false;
            SelectedViewMode = ViewMode.TileView;
        }

        protected void HideReviewButton()
        {
            IsReviewButtonVisible = false;
        }

        protected void ShowReviewButton()
        {
            IsReviewButtonVisible = true;
        }

        protected async Task OnReviewClicked()
        {
            await GridView.OnReviewClicked();
        }

        #endregion
    }
}