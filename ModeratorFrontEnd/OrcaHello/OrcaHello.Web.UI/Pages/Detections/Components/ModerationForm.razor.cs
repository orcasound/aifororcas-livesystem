
using OrcaHello.Web.UI.Models.DetectionViews;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using static System.Net.WebRequestMethods;

namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    [Authorize(Roles = "Moderator")]
    public partial class ModerationForm
    {
        [Inject]
        HttpClient HttpClient { get; set; }

        [Parameter]
        public DetectionItemView DetectionItemView { get; set; }

        private JsonSerializerOptions defaultJsonSerializerOptions => new JsonSerializerOptions() { PropertyNameCaseInsensitive = true };

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

            // Specify the scopes required by the API
            string[] scopes = new string[] {"user.read"};
            var authenticationScheme = "Cookies";

            // Get the access token from the ITokenAcquisition service
            //string accessToken = await TokenAcquisition.GetAccessTokenForUserAsync(scopes, authenticationScheme: authenticationScheme);

            //// Use the access token to call the API
            //HttpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

            //    var response = await HttpClient.GetAsync("https://localhost:5001/api/home/moderator");

            //if (response.IsSuccessStatusCode)
            //{
            //    var data = await response.Content.ReadAsStringAsync();

            //}



            var httpResponseMessage = await HttpClient.GetAsync("https://localhost:5001/api/home/moderator");

            if (httpResponseMessage.IsSuccessStatusCode)
            {
                var responseString = await httpResponseMessage.Content.ReadAsStringAsync();
            }
            else
            {
            }


        }

        // Get an access token for the API scope
        //var accessToken = await TokenAcquisition.GetAccessTokenForUserAsync(new[] { "api://8a17ae31-03c1-455a-b1ef-023bd62d79d6/API.Access" });

        //            // Create an HttpClient with the access token in the Authorization header
        //            var httpClient = new HttpClient();
        //            httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

        //            // Call the API and process the response
        //            var response = await httpClient.GetAsync("https://localhost:5001/api/home/moderator");
        //            var content = await response.Content.ReadAsStringAsync();
        //            // Do something with content

        //    }

        }
    }
