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
        return File.ReadAllText(Path.Combine(_solutionDirectory, "NotificationSystemTests", "TestData", filename));
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
            string functionHostDirectory = Path.Combine(solutionDirectory ?? "", "NotificationSystem");
            return functionHostDirectory;
        }
    }

    /// <summary>
    /// This test updates a Cosmos DB item to trigger the Azure Function and verifies that it runs successfully.
    /// Since the Azure Function runs out of process, the Orcasite API is not mocked
    /// so this relies on environment variable configuration for the post to succeed.
    /// Such configuration should use beta.orcasound.net (the default).
    /// </summary>
    /// <returns></returns>
    [Fact(Timeout = 60000)] // 60 seconds max
    [Trait("Category", "Cosmos")]
    public async Task UpdateCosmosDb()
    {
        // Use the Azure Cosmos DB Emulator connection string for local testing.
        using CosmosClient client = new(
            accountEndpoint: "https://localhost:8081/",
            authKeyOrResourceToken: "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==");

        Microsoft.Azure.Cosmos.Database database = await client.CreateDatabaseIfNotExistsAsync(
            id: "predictions",
             throughput: 400);

        Container metadataContainer = await database.CreateContainerIfNotExistsAsync(
            id: "metadata",
            partitionKeyPath: "/source_guid"
        );

        Container leasesContainer = await database.CreateContainerIfNotExistsAsync(
            id: "leases",
            partitionKeyPath: "/id");

        // Start Azure function process from the function host directory.
        string workingDirectory = FunctionHostDirectory;
        Console.WriteLine($"Function host working directory: {workingDirectory}");
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                WorkingDirectory = workingDirectory,
                FileName = "func",
                Arguments = "start",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };
        int postsAttempted = 0;
        int postsSucceeded = 0;
        process.OutputDataReceived += (sender, args) =>
        {
            string data = args.Data ?? "";
            if (data.Contains("Executing 'PostToOrcasite'"))
            {
                postsAttempted++;
            }

            if (data.Contains("posted successfully"))
            {
                postsSucceeded++;
            }
        };
        bool processStarted = false;
        try
        {
            processStarted = process.Start();
            Assert.True(processStarted);
            process.BeginOutputReadLine();

            var item = JsonConvert.DeserializeObject<dynamic>(_sampleOrcaHelloDetection);
            if (item == null)
            {
                return;
            }

            // Randomize a value to ensure it will be updated.
            var random = new Random();
            item.whaleFoundConfidence = random.NextDouble() * 30 + 50;

            int oldPostsAttempted = postsAttempted;
            int oldPostsSucceeded = postsSucceeded;

            dynamic? result = await metadataContainer.UpsertItemAsync(item);
            int httpStatusCode = (int)result.StatusCode;
            Console.WriteLine($"Cosmos DB Emulator returned status: {httpStatusCode}");
            Assert.True(httpStatusCode >= 200 && httpStatusCode < 300, $"Cosmos DB update failed with status {httpStatusCode}");

            // Wait up to 20 seconds for the Azure function to execute.
            const int maxSeconds = 20;
            for (int seconds = 0; seconds < maxSeconds && postsSucceeded == oldPostsSucceeded; seconds++)
            {
                Console.Write(".");
                await Task.Delay(1000); // Wait one second before checking again.
            }

            // Verify it ran.
            Assert.True(postsAttempted > oldPostsAttempted, $"Incorrect posts attempted: {postsAttempted}");
            Assert.True(postsSucceeded > oldPostsSucceeded, $"Incorrect posts succeeded: {postsSucceeded}");
        }
        finally
        {
            if (processStarted && !process.HasExited)
            {
                // Clean up: kill the function host process.
                process.Kill(true); // true = kill entire process tree
                process.WaitForExit();
            }
        }
    }
}
