using Microsoft.Azure.Documents;
using Microsoft.Extensions.Logging;
using Moq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NotificationSystem.Models;
using RichardSzalay.MockHttp;
using System.Net;
using System.Text.Json;

namespace NotificationSystem.Tests.Common
{
    public class OrcasiteTestHelper
    {
        private static string _solutionDirectory;

        /// <summary>
        /// Find the solution directory.
        /// </summary>
        /// <returns>Directory, or null if not found</returns>
        public static string? FindSolutionDirectory()
        {
            if (_solutionDirectory != null)
            {
                return _solutionDirectory;
            }
            string? currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
            while (currentDirectory != null)
            {
                string path = Path.Combine(currentDirectory, "NotificationSystem.sln");
                if (File.Exists(path))
                {
                    _solutionDirectory = currentDirectory;
                    return currentDirectory;
                }

                currentDirectory = Directory.GetParent(currentDirectory)?.FullName;
            }
            return null;
        }

        /// <summary>
        /// Get the contents of a TestData file as a string.
        /// </summary>
        /// <param name="filename">Name of file to load</param>
        /// <returns>String contents</returns>
        private static string GetStringFromFile(string filename)
        {
            string solutionDirectory = FindSolutionDirectory() ?? throw new Exception("Could not find solution directory");
            return File.ReadAllText(Path.Combine(solutionDirectory, "TestData", filename));
        }

        public static OrcasiteHelper GetMockOrcasiteHelper(ILogger<OrcasiteHelper> logger)
        {
            var mockHttp = new MockHttpMessageHandler();
            string sampleOrcasiteFeeds = GetStringFromFile("OrcasiteFeeds.json");
            string sampleOrcasitePostDetectionResponse = GetStringFromFile("OrcasitePostDetectionResponse.json");

            // Mock the GET request to fetch feeds.
            mockHttp.When(HttpMethod.Get, "https://beta.orcasound.net/api/json/feeds?fields%5Bfeed%5D=id%2Cname%2Cnode_name%2Cslug%2Clocation_point%2Cintro_html%2Cimage_url%2Cvisible%2Cbucket%2Cbucket_region%2Ccloudfront_url%2Cdataplicity_id%2Corcahello_id")
                    .Respond("application/json", sampleOrcasiteFeeds);

            // Mock the POST request to create a detection.
            mockHttp.When(HttpMethod.Post, "https://beta.orcasound.net/api/json/detections?fields%5Bdetection%5D=id%2Csource_ip%2Cplaylist_timestamp%2Cplayer_offset%2Clistener_count%2Ctimestamp%2Cdescription%2Cvisible%2Csource%2Ccategory%2Ccandidate_id%2Cfeed_id")
                    //.WithContent("{\"key\":\"value\"}") // Optional: match request body
                    .Respond(HttpStatusCode.Created, "application/json", sampleOrcasitePostDetectionResponse);

            var httpClient = mockHttp.ToHttpClient();
            var orcasiteHelper = new OrcasiteHelper(logger, httpClient);
            return orcasiteHelper;
        }

        public static List<JsonElement> GetSampleOrcaHelloDetections()
        {
            string sampleOrcaHelloDetection = GetStringFromFile("OrcaHelloDetection.json");
            JsonElement testDocument = JsonDocument.Parse(sampleOrcaHelloDetection).RootElement;
            var documents = new List<JsonElement> { testDocument };
            return documents;
        }
    }
}
