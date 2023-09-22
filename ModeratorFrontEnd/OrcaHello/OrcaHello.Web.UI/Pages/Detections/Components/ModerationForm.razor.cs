using OrcaHello.Web.UI.Models.DetectionViews;

namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    [Authorize(Roles = "Moderator")]
    public partial class ModerationForm
    {
        //[Inject]
        //ITokenAcquisition TokenAcquisition { get; set; }

        //[Inject]
        //MicrosoftIdentityConsentAndConditionalAccessHandler ConsentHandler { get; set; }



        [Parameter]
        public DetectionItemView DetectionItemView { get; set; }



        private ModerateDetection moderateDetection = new();

        protected override void OnInitialized()
        {
            moderateDetection = new ModerateDetection
            {
                Id = moderateDetection.Id,
                State = DetectionItemView.State,
                Comments = DetectionItemView.Comments,
                Tags = DetectionItemView.Tags,
                InterestLabel = DetectionItemView.InterestLabel
            };
        }

        //private string Moderator { get; set; }
        //private DateTime? Moderated { get; set; }

        private async Task Submit()
        {
            //// Do something with the form data
            //string apiResponse = string.Empty;
            //try
            //{
            //    // Get an access token for the API scope
            //    var accessToken = await TokenAcquisition.GetAccessTokenForUserAsync(new[] { "api-scope" });

            //    // Create an HttpClient with the access token in the Authorization header
            //    var httpClient = new HttpClient();
            //    httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

            //    var jsonContent = new StringContent(JsonConvert.SerializeObject(moderateDetection));

            //    // Set the ContentType property to application/json.
            //    jsonContent.Headers.ContentType = new MediaTypeHeaderValue("application/json");


            //    // Call the API and process the response
            //    var response = await httpClient.PutAsync($"https://localhost:5001/{moderateDetection.Id}/moderator", jsonContent);
            //    apiResponse = await response.Content.ReadAsStringAsync();
            //}
            //catch (Exception ex)
            //{
            //    // Handle any exceptions that require user interaction
            //    ConsentHandler.HandleException(ex);
            //}
        }
    }
}
