using System;

namespace OrcaHello.Web.UI.Pages.Tags.Components
{
    /// <summary>
    /// Code-behind for the ReplaceTagComponent.razor page, providing functionality for replacing tags.
    /// </summary>
    public partial class ReplaceTagComponent
    {
        /// <summary>
        /// Injected service for interacting with tag-related data.
        /// </summary>
        [Inject]
        public ITagViewService ViewService { get; set; } = null!;

        /// <summary>
        /// Represents the TagItemView for which the tag is being replaced.
        /// </summary>
        [Parameter]
        public TagItemView Item { get; set; } = null!;

        /// <summary>
        /// Event callback to notify the parent component when the dialog is closed.
        /// </summary>
        [Parameter]
        public EventCallback<bool> OnCloseClicked { get; set; }

        /// <summary>
        /// Holds the content of the new tag being entered.
        /// </summary>
        // Content being edited
        protected string NewTag = string.Empty;

        /// <summary>
        /// Holds the validation message for the new tag.
        /// </summary>
        protected string NewTagValidationMessage = string.Empty; // Error message holder

        /// <summary>
        /// Handles the replacement of the old tag with the new one.
        /// </summary>
        protected async Task OnReplaceClicked()
        {
            // Check if the new tag is empty, and set a validation message if true
            if (string.IsNullOrEmpty(NewTag))
                NewTagValidationMessage = "You must enter a new tag in order to replace the old one.";

            // Create a ReplaceTagRequest with old and new tag information
            ReplaceTagRequest request = new()
            {
                OldTag = Item.Tag,
                NewTag = NewTag
            };

            try
            {
                // Attempt to replace the tag using the ViewService
                var response = await ViewService.ReplaceTagAsync(request);

                // Check if the tag replacement was successful and report accordingly
                if (response.MatchingTags == response.ProcessedTags)
                    ReportSuccess("Success", "Tag was successfully replaced in all detections.");
                else
                    ReportError("Failure", "Tag was not replaced in one or more detections.");

                // Invoke the OnCloseClicked event with a parameter indicating success
                await OnCloseClicked.InvokeAsync(true);
            }
            catch (Exception exception)
            {
                // Report any errors during tag replacement
                LogAndReportUnknownException(exception);
            }
        }

        /// <summary>
        /// Handles the cancellation of tag replacement, notifying the parent component that the dialog is closed.
        /// </summary>
        protected async Task OnCancelClicked()
        {
            // Invoke the OnCloseClicked event with a parameter indicating cancellation
            await OnCloseClicked.InvokeAsync(false);
        }
    }
}