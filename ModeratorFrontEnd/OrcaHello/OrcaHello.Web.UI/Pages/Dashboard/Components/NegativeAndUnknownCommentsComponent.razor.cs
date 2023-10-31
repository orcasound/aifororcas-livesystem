namespace OrcaHello.Web.UI.Pages.Dashboard.Components
{
    public partial class NegativeAndUnknownCommentsComponent
    {
        [Inject]
        public IMetricsViewService ViewService { get; set; } = null!;

        [Parameter]
        public CommentStateView StateView { get; set; } = null!;

        [Parameter]
        public EventCallback OnToggleOpen { get; set; }

        protected CommentItemView ActiveItem = null!; // Holder for item being edited or played

        #region component buttons

        protected async Task ToggleNegativeAndUnknownComments()
        {
            // Getting ready to expand this component, so make sure all the other ones are closed

            if (!StateView.IsExpanded)
            {
                await OnToggleOpen.InvokeAsync();
            }

            StateView.Toggle();

            if (!StateView.Items.Any())
            {
                await LoadNegativeAndUnknownCommentsAsync();
            }

        }

        protected async Task OnLoadMoreNegativeAndUnknownCommentsClicked()
        {
            StateView.Page++;
            await LoadNegativeAndUnknownCommentsAsync();
        }

        #endregion

        #region grid audio events

        protected async Task OnPlayAudioClicked(CommentItemView item)
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

        protected async Task OnStopAudioClicked(CommentItemView item)
        {
            await JSRuntime.InvokeVoidAsync("StopGridAudioPlayback", item.Id, item.AudioUri);

            ActiveItem.IsCurrentlyPlaying = false;
            await InvokeAsync(StateHasChanged);
            ActiveItem = null!;
        }

        #endregion

        #region data loaders


        protected async Task LoadNegativeAndUnknownCommentsAsync()
        {
            StateView.IsLoading = true;

            PaginatedCommentsByDateRequest request = new()
            {
                Page = StateView.Page,
                PageSize = StateView.PageSize,
                FromDate = StateView.FromDate,
                ToDate = StateView.ToDate,
            };

            try
            {
                var result = await ViewService
                    .RetrieveFilteredNegativeAndUnknownCommentsAsync(request);

                // Update the Data property
                if (result.CommentItemViews.Any())
                    StateView.Items.AddRange(result.CommentItemViews);

                // Update the count
                StateView.Count = result.Count;
            }
            catch (Exception ex)
            {
                ReportError("Trouble loading negative and unknown Comments data", ex.Message);
            }

            StateView.IsLoading = false;
        }

        #endregion
    }
}
