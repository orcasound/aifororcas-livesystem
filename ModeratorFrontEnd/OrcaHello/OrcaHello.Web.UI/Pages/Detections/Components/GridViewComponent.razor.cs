namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class GridViewComponent
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Inject]
        public IAccountService AccountService { get; set; } = null!;

        [Parameter]
        public Filters Filters { get; set; } = null!;

        [Parameter]
        public EventCallback OnShowReviewButton { get; set; }

        RadzenDataGrid<DetectionItemView> DetectionDataGrid = null!; // Reference to the grid component

        protected bool IsLoading = false; // Flag indicating data is being loaded into the grid

        protected IEnumerable<DetectionItemView> DetectionItemViews = null!; // List of items currently in the grid

        IList<DetectionItemView> SelectedDetectionItemViews = new List<DetectionItemView>(); // Which items have been selected in the grid

        protected Filters CurrentlySetFilters = new(); // The currently "applied" filters; so they don't get updated until Apply is clicked

        protected int TotalDetectionCount = 0; // The total number of records for the applied filter

        protected DetectionItemView ActiveItem = null!; // Holder for item being edited or played

        protected bool IsInlineEditing = false; // Flag indicating an inline edit is under way

        protected string Moderator = string.Empty; // The name of the current moderator, if logged in

        protected List<string> AvailableTags = new(); // All unique tags in the system for picklists

        #region lifecycle events

        protected override async Task OnInitializedAsync()
        {
            Moderator = await AccountService.GetUserName();
            AvailableTags = await ViewService.RetrieveAllTagsAsync();

            await OnShowReviewButton.InvokeAsync();
        }

        protected override async Task OnParametersSetAsync()
        {
            if (CurrentlySetFilters != Filters)
            {
                CurrentlySetFilters = Filters;

                if (DetectionDataGrid?.Data != null)
                    await DetectionDataGrid.Reload();
            }
        }

        #endregion

        #region bulk review

        protected void OnAllItemsChanged(bool? value)
        {
            SelectedDetectionItemViews = value == true ? DetectionItemViews.ToList() : null!;
        }

        public async Task OnReviewClicked()
        {
            if (IsInlineEditing)
                ReportError("Inline Edit",
                    "You must complete your inline edit of data before you can perform a bulk review.");

            else if (SelectedDetectionItemViews == null || !SelectedDetectionItemViews.Any())
                ReportError("Select a Record",
                    "You must select one or more records to bulk review.");

            else
            {
                List<string> selectedIds = SelectedDetectionItemViews.Select(x => x.Id).ToList();

                await DialogService.OpenAsync<BulkReviewComponent>(
                   "Review Candidates",
                   new Dictionary<string, object>() {
                       { "SelectedIds", selectedIds },
                       { "Moderator", Moderator },
                       { "AvailableTags", AvailableTags },
                       { "OnCloseClicked", EventCallback.Factory.Create<bool>(this, HandleBulkReviewClosed) }
                   },
                   new DialogOptions() { Width = "450px", Height = "525px", Resizable = false, Draggable = true }
                );
            }
        }

        public async Task HandleBulkReviewClosed(bool reload)
        {
            DialogService.Close();

            if (reload)
                await ReloadData();
        }

        #endregion

        #region inline editing events

        public async Task OnInlineEditRowClicked(DetectionItemView item)
        {
            IsInlineEditing = true;

            ActiveItem = new DetectionItemView
            {
                State = item.State,
                Comments = item.Comments,
                Tags = item.Tags
            };

            await DetectionDataGrid.EditRow(item);
        }

        public async Task OnSaveRowClicked(DetectionItemView item)
        {
            if (string.IsNullOrWhiteSpace(item.State) || item.State == DetectionState.Unreviewed.ToString())
                ReportError("Select Call State",
                    "A Call State (Yes, No, Don't Know) must be selected in order to save this record.");

            else
            {
                var response = await ViewService.ModerateDetectionsAsync(
                    new List<string> { item.Id },
                    item.State,
                    Moderator,
                    item.Comments,
                    item.Tags);

                IsInlineEditing = false;

                if (response != null)
                {
                    if (response.IdsSuccessfullyUpdated.Count > 0)
                        ReportSuccess("Success",
                            "Record was successfully updated (and may disappear from this list).");

                    else if (response.IdsNotFound.Count > 0)
                        ReportError("Not Found",
                            "Record not found in the database and could not be updated.");
                    else
                        ReportError("Failure",
                            "An unknow error occurred while updating the record.");
                }

                await ReloadData();
            }
        }

        public void OnCancelEditClicked(DetectionItemView item)
        {
            item.State = ActiveItem.State;
            item.Comments = ActiveItem.Comments;
            item.Tags = ActiveItem.Tags;

            ActiveItem = null!;
            IsInlineEditing = false;

            DetectionDataGrid.CancelEditRow(item);
        }

        #endregion

        #region grid audio events

        protected async Task OnPlayAudioClicked(DetectionItemView item)
        {
            if (ActiveItem is null)
            {
                ActiveItem = item;
                ActiveItem.IsCurrentlyPlaying = true;
            }
            else
            {
                await JSRuntime.InvokeVoidAsync("StopGridAudioPlayback");
                ActiveItem.IsCurrentlyPlaying = false;
                await InvokeAsync(StateHasChanged);
                ActiveItem = item;
                ActiveItem.IsCurrentlyPlaying = true;
            }

            CustomJSEventHelper helper = new CustomJSEventHelper(OnDonePlaying);
            DotNetObjectReference<CustomJSEventHelper> reference =
                DotNetObjectReference.Create(helper);

            await JSRuntime.InvokeVoidAsync("StartGridAudioPlayback", item.AudioUri, reference);
        }

        protected async Task OnDonePlaying(EventArgs args)
        {
            await JSRuntime.InvokeVoidAsync("StopGridAudioPlayback");

            ActiveItem.IsCurrentlyPlaying = false;
            await InvokeAsync(StateHasChanged);
            ActiveItem = null!;
        }

        protected async Task OnStopAudioClicked(DetectionItemView item)
        {
            await JSRuntime.InvokeVoidAsync("StopGridAudioPlayback", item.Id, item.AudioUri);

            ActiveItem.IsCurrentlyPlaying = false;
            await InvokeAsync(StateHasChanged);
            ActiveItem = null!;
        }

        #endregion

        #region helpers

        async Task ReloadData()
        {
            DetectionDataGrid.Data = null;
            await DetectionDataGrid.Reload();
        }

        private async Task LoadData(LoadDataArgs args)
        {
            IsLoading = true;
            await InvokeAsync(StateHasChanged);

            // This sets the initial value to Range (if populated) or All (if not)
            var fromDate = Filters.StartDateTime.HasValue ? Filters.StartDateTime.Value : new DateTime(1970, 1, 1);
            var toDate = Filters.EndDateTime.HasValue ? Filters.EndDateTime.Value : DateTime.UtcNow;

            if (Filters.Timeframe != Timeframe.All && Filters.Timeframe != Timeframe.Range)
                fromDate = toDate.Adjust(Filters.Timeframe);

            int skip = args.Skip.HasValue ? args.Skip.Value : 1;
            int top = args.Top.HasValue ? args.Top.Value : 10;

            DetectionFilterAndPagination filterAndPagination = new()
            {
                Page = (skip / top) + 1,
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

        #endregion
    }
}
