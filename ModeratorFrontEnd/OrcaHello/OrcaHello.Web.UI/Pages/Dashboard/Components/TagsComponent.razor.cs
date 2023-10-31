namespace OrcaHello.Web.UI.Pages.Dashboard.Components
{
    public partial class TagsComponent
    {
        [Inject]
        public IMetricsViewService ViewService { get; set; } = null!;

        [Parameter]
        public TagStateView StateView { get; set; } = null!;

        [Parameter]
        public EventCallback OnToggleOpen { get; set; }

        protected TagDetectionStateView TagDetectionsState = new();

        protected DetectionItemView ActiveItem = null!; // Holder for item being edited or played

        #region component buttons

        protected async Task ToggleTags()
        {
            // Getting ready to expand this component, so make sure all the other ones are closed

            if (!StateView.IsExpanded)
            {
                await OnToggleOpen.InvokeAsync();
            }

            StateView.Toggle();

            if (!StateView.Items.Any())
            {
                await LoadTagsAsync();
            }
        }

        protected async Task OnTagExpanded(object value)
        {
            int selectedTagIndex = (int)value;

            TagDetectionsState.SelectedTagReset(StateView.Items[selectedTagIndex]);

            await LoadTagDetectionsAsync();
        }

        protected async Task OnLoadMoreTagDetectionsClicked()
        {
            TagDetectionsState.Page++;

            await LoadTagDetectionsAsync();
        }

        #endregion

        #region data loaders

        protected async Task LoadTagsAsync()
        {
            StateView.IsLoading = true;

            TagsByDateRequest request = new()
            {
                FromDate = StateView.FromDate,
                ToDate = StateView.ToDate,
            };

            try
            {
                StateView.Items = await ViewService.RetrieveFilteredTagsAsync(request);
                StateView.Count = StateView.Items.Count;
            }
            catch(Exception ex)
            {
                ReportError("Trouble loading Tags data", ex.Message);
            }

            StateView.IsLoading = false;

            await InvokeAsync(StateHasChanged);
        }

        protected async Task LoadTagDetectionsAsync()
        {
            TagDetectionsState.IsLoading = true;

            PaginatedDetectionsByTagAndDateRequest request = new()
            {
                Page = TagDetectionsState.Page,
                PageSize = TagDetectionsState.PageSize,
                FromDate = StateView.FromDate,
                ToDate = StateView.ToDate,
                Tag = TagDetectionsState.Tag
            };

            try
            {

                var result = await ViewService
                    .RetrieveFilteredDetectionsForTagsAsync(request);

                // Update the Data property
                if (result.DetectionItemViews.Any())
                    TagDetectionsState.Items.AddRange(result.DetectionItemViews);

                // Update the count
                TagDetectionsState.Count = result.Count;

            }
            catch(Exception ex)
            {
                ReportError("Trouble loading Tag Detections data", ex.Message);
            }

            TagDetectionsState.IsLoading = false;
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
    }
}
