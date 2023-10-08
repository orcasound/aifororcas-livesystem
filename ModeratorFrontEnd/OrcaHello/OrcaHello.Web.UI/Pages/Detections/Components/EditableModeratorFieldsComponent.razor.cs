namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class EditableModeratorFieldsComponent
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; }

        [Inject]
        public IAccountService AccountService { get; set; }

        [Parameter]
        public DetectionItemView Item { get; set; } = new();

        protected async Task OnSubmitClicked()
        {
            // TODO: How to get Moderator if not already in the item?

            string moderator = await AccountService.GetUsername();

            var response = await ViewService.ModerateDetectionsAsync(
                new List<string> { Item.Id },
                Item.State,
                moderator,
                Item.Comments,
                Item.Tags);

            // TODO: Display the toast acknowledgement
            // TODO: Force the reload of the view (probably after an EventTrigger on the parent component)
        }
    }
}
