namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class AddTagComponent
    {
        // Declare a parameter to receive the AvailableTags list from the parent component
        [Parameter]
        public List<string> AvailableTags { get; set; }

        string NewTag;

        protected void AddTag()
        {
            // Check if the new tag is not empty or already in the list
            if (!string.IsNullOrEmpty(NewTag))
            {
                // Close the dialog and return the new tag
                DialogService.Close(NewTag);
            }
        }
    }
}
