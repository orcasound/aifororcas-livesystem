namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class ReviewCandidatesComponent
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; }

        [Inject]
        public IAccountService AccountService { get; set; }


        [Parameter]
        public List<string> SelectedIds { get; set; }

        protected string Tags = string.Empty;
        protected string Comments = string.Empty;
        protected string State = string.Empty;

        protected async Task OnSaveAndSubmitClicked()
        {
            // TODO: Data Validation

            string moderator = await AccountService.GetUsername();

            var response = await ViewService.ModerateDetectionsAsync(
                SelectedIds,
                State,
                moderator,
                Comments,
                Tags);

            // TODO: Close Dialog on finish
            // TODO: Redraw after save and submit
        }

        protected async Task OnSaveClicked()
        {

        }

        protected async Task OnCancelClicked()
        {

        }
    }
}
