//namespace OrcaHello.Web.UI.Pages.Detections.Components
//{
//    public partial class TileModeComponent
//    {
//        [Inject]
//        public IDetectionViewService ViewService { get; set; }

//        protected IEnumerable<DetectionItemView> DetectionItemViews;
//        protected int TotalDetectionCount = 0;
//        protected bool IsLoading = false;

//        RadzenDataList<DetectionItemView> DetectionDataList;
//        RadzenDataGrid<DetectionItemView> DetectionDataGrid;

//        IList<DetectionItemView> selectedDetectionItemViews = new List<DetectionItemView>();

//        protected async Task OnReviewClicked()
//        {
//            List<string> selectedIds = selectedDetectionItemViews.Select(x => x.Id).ToList();

//            await DialogService.OpenAsync<ReviewCandidatesComponent>($"Review Candidates",
//                new Dictionary<string, object>() { { "SelectedIds", selectedIds } },
//                new DialogOptions() { Width = "450px", Height = "430px", Resizable = false, Draggable = true });

//            // TODO: a reload after closed?
//        }

//        async Task ReloadData()
//        {
//            if (ViewMode == 0)
//            {
//                DetectionDataList.Data = null;
//                await DetectionDataList.Reload();
//            }
//            else
//            {
//                DetectionDataGrid.Data = null;
//                await DetectionDataGrid.Reload();
//            }
//        }

//        private async Task LoadData(LoadDataArgs args)
//        {
//            IsLoading = true;
//            await InvokeAsync(StateHasChanged);

//            var state = DetectionStateValue.HasValue ?
//                DetectionStateOptions[DetectionStateValue.Value].ValueText : string.Empty;

//            var sortBy = SortByValue.HasValue ?
//                SortByOptions[SortByValue.Value].ValueText : string.Empty;

//            var isDescending = SortOrderValue.HasValue ?
//                SortOrderOptions[SortOrderValue.Value].ValueText == "ASC" ? false : true : true;

//            var locationName = LocationValue.HasValue && LocationValue.Value < LocationOptions.Count - 1 ?
//                LocationOptions[LocationValue.Value].Text : string.Empty;

//            // This sets the initial value to all
//            var fromDate = StartDateTime.HasValue ? StartDateTime.Value : new DateTime(1970, 1, 1);
//            var toDate = EndDateTime.HasValue ? EndDateTime.Value : DateTime.UtcNow;

//            if (TimeframeValue.HasValue)
//            {
//                switch (TimeframeOptions[TimeframeValue.Value].ValueText)
//                {
//                    case "30min":
//                        fromDate = toDate.AddMinutes(-30);
//                        break;
//                    case "3hrs":
//                        fromDate = toDate.AddHours(-3);
//                        break;
//                    case "6hrs":
//                        fromDate = toDate.AddHours(-6);
//                        break;
//                    case "24hrs":
//                        fromDate = toDate.AddHours(-24);
//                        break;
//                    case "7dys":
//                        fromDate = toDate.AddDays(-7);
//                        break;
//                    case "30dys":
//                        fromDate = toDate.AddDays(-30);
//                        break;
//                }
//            }

//            int pageSize = MaxRecordsValue.HasValue ? MaxRecordsValue.Value : 10;



//            DetectionFilterAndPagination filterAndPagination = new()
//            {
//                Page = (args.Skip.Value / args.Top.Value) + 1,
//                PageSize = pageSize,
//                State = state,
//                SortBy = sortBy,
//                IsDescending = isDescending,
//                FromDate = fromDate,
//                ToDate = toDate,
//                Location = locationName
//            };

//            var result = await ViewService.RetrieveFilteredAndPaginatedDetectionItemViewsAsync(filterAndPagination);
//            // Update the Data property
//            DetectionItemViews = result.DetectionItemViews;
//            // Update the count
//            TotalDetectionCount = result.Count;

//            IsLoading = false;
//            await InvokeAsync(StateHasChanged);
//        }
//    }
//}
