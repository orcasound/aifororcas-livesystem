﻿namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class BulkReviewComponent
    {
        [Inject]
        public IDetectionViewService ViewService { get; set; } = null!;

        [Parameter]
        public EventCallback<bool> OnCloseClicked { get; set; }

        [Parameter]
        public List<string> SelectedIds { get; set; } = new();

        [Parameter]
        public string Moderator { get; set; } = string.Empty;

        [Parameter]
        public List<string> AvailableTags { get; set; } = new();

        // Content being edited
        protected string Tags = string.Empty;
        protected string Comments = string.Empty;
        protected string State = string.Empty;


        protected string StateValidationMessage = string.Empty; // Error message holder

        // Working values for tags being selected and entered
        protected List<string> SelectedTags = new();
        protected string EnteredTags = string.Empty;

        protected async Task OnSubmitClicked()
        {
            if (string.IsNullOrEmpty(State))
                StateValidationMessage = "You must indicate whether a whale call was heard, not heard, or undetermined for the selected candidates.";

            else
            {
                SelectedTags.AddRange(EnteredTags.Split(new char[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries));
                var trimmedList = SelectedTags.Select(s => s.Trim());
                Tags = string.Join(",", trimmedList);

                var response = await ViewService.ModerateDetectionsAsync(
                    SelectedIds,
                    State,
                    Moderator,
                    Comments,
                    Tags);

                if (response.IdsToUpdate.Count == response.IdsSuccessfullyUpdated.Count)
                    ReportSuccess("Success",
                        "All records successfully updated (and may disappear from this list).");

                else
                    ReportError("Failure",
                        "One or more records were not updated.");

                await OnCloseClicked.InvokeAsync(true);
            }
        }

        protected async Task OnCancelClicked()
        {
            await OnCloseClicked.InvokeAsync(false);
        }
    }
}