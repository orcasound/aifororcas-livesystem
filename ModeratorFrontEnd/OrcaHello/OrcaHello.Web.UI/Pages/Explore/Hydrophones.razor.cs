namespace OrcaHello.Web.UI.Pages.Explore
{
    // Provides the logic and data for the Hydrophones.razor page that shows the hydrophone information to the user.
    // It retrieves the data from the ViewService and displays it in a grid and a map.It also handles the loading
    // and error states. It does not need code coverage testing.
    [ExcludeFromCodeCoverage]
    public partial class Hydrophones
    {
        [Inject]
        public IHydrophoneViewService ViewService { get; set; } = null!;

        // Store the data retrieved from the ViewService and display it to the user.
        protected List<HydrophoneItemView> HydrophoneItemViews = null!;

        // Indicate whether the data is being loaded or not.
        protected bool IsLoading = false;

        // Control the zoom level of the map component that shows the hydrophone locations.
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

        #region data loaders

        // Loads the hydrophone data from the ViewService and assigns it to the HydrophoneItemViews property.
        // It also handles the loading state and any exceptions that may occur during the process.
        private async Task LoadDataAsync()
        {
            try
            {
                // Set the IsLoading property to true and the HydrophoneItemViews property to null.
                // This will make the page show a loading indicator and clear any existing data.
                IsLoading = true;
                HydrophoneItemViews = null!;
                // Invoke the StateHasChanged method to notify the component that its state has changed and trigger a re-render.
                await InvokeAsync(StateHasChanged);

                // Call the RetrieveAllHydrophoneViewsAsync method from the ViewService and assign the result to the HydrophoneItemViews property.
                HydrophoneItemViews = await ViewService.RetrieveAllHydrophoneViewsAsync();

                // Set the IsLoading property to false.
                // This will make the page hide the loading indicator and show the data.
                IsLoading = false;
                // Invoke the StateHasChanged method again to update the component with the new data.
                await InvokeAsync(StateHasChanged);
            }
            catch (Exception exception)
            {
                // Set the IsLoading property to false.
                IsLoading = false;
                // Make the page show 'no records found' on any load exception by assigning an empty list to the HydrophoneItemViews property.
                HydrophoneItemViews = new();
                // Call the LogAndReportUnknownException method with the exception as a parameter.
                // This method will log the exception using the LoggingUtilities and report it to the user using the NotificationService.
                LogAndReportUnknownException(exception);
            }
        }

        #endregion
    }
}
