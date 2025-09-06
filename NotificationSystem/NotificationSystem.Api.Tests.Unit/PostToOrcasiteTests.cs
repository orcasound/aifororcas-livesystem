using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Documents;
using Microsoft.Extensions.Logging;
using Moq;
using Newtonsoft.Json;
using NotificationSystem;
using NotificationSystem.Models;
using RichardSzalay.MockHttp;
using System.Diagnostics;
using System.Net;

public class PostToOrcasiteTests
{
    private string _solutionDirectory;
    private string _sampleOrcaHelloDetection;
    private string _sampleOrcasiteFeeds;
    private string _sampleOrcasitePostDetectionResponse;

    public PostToOrcasiteTests()
    {
        _solutionDirectory = FindSolutionDirectory() ?? throw new Exception("Could not find solution directory");
        _sampleOrcaHelloDetection = GetStringFromFile("OrcaHelloDetection.json");
        _sampleOrcasiteFeeds = GetStringFromFile("OrcasiteFeeds.json");
        _sampleOrcasitePostDetectionResponse = GetStringFromFile("OrcasitePostDetectionResponse.json");
    }

    /// <summary>
    /// Find the solution directory.
    /// </summary>
    /// <returns>Directory, or null if not found</returns>
    string? FindSolutionDirectory()
    {
        string? currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
        while (currentDirectory != null)
        {
            string path = Path.Combine(currentDirectory, "NotificationSystem.sln");
            if (File.Exists(path))
            {
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
    private string GetStringFromFile(string filename)
    {
        return File.ReadAllText(Path.Combine(_solutionDirectory, "NotificationSystem.Api.Tests.Unit", "TestData", filename));
    }

    /// <summary>
    /// This test mocks the Orcasite API and verifies that a detection can be posted successfully.
    /// </summary>
    /// <returns></returns>
    [Fact]
    public async Task PostToOrcasite_ProcessDocumentsAsync()
    {
        var mockLogger = new Mock<ILogger>();

        var mockHttp = new MockHttpMessageHandler();

        // Mock the GET request to fetch feeds.
        mockHttp.When(HttpMethod.Get, "https://beta.orcasound.net/api/json/feeds?fields%5Bfeed%5D=id%2Cname%2Cnode_name%2Cslug%2Clocation_point%2Cintro_html%2Cimage_url%2Cvisible%2Cbucket%2Cbucket_region%2Ccloudfront_url%2Cdataplicity_id%2Corcahello_id")
                .Respond("application/json", _sampleOrcasiteFeeds);

        // Mock the POST request to create a detection.
        mockHttp.When(HttpMethod.Post, "https://beta.orcasound.net/api/json/detections?fields%5Bdetection%5D=id%2Csource_ip%2Cplaylist_timestamp%2Cplayer_offset%2Clistener_count%2Ctimestamp%2Cdescription%2Cvisible%2Csource%2Ccategory%2Ccandidate_id%2Cfeed_id")
              //.WithContent("{\"key\":\"value\"}") // Optional: match request body
                .Respond(HttpStatusCode.Created, "application/json", _sampleOrcasitePostDetectionResponse);

        var httpClient = mockHttp.ToHttpClient();
        var orcasiteHelper = new OrcasiteHelper(mockLogger.Object, httpClient);

        var document = JsonConvert.DeserializeObject<Microsoft.Azure.Documents.Document>(_sampleOrcaHelloDetection);
        if (document == null)
        {
            return;
        }
        var documents = new List<Document> { document };

        // Process it like the Azure function would.
        await PostToOrcasite.ProcessDocumentsAsync(documents, orcasiteHelper, mockLogger.Object);
    }

    /// <summary>
    /// Directory in which the Azure Function host can be started.
    /// </summary>
    string FunctionHostDirectory
    {
        get
        {
            string? solutionDirectory = FindSolutionDirectory();
            string functionHostDirectory = Path.Combine(solutionDirectory ?? "", "NotificationSystem.Api");
            return functionHostDirectory;
        }
    }
}
