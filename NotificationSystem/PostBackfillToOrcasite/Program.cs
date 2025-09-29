// This is a console tool to post the history
// of OrcaHello detections to Orcasite, for data
// analysis purposes.

using Microsoft.Azure.Cosmos;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using NotificationSystem;
using NotificationSystem.Models;
using System.Text.Json;

namespace PostBackfillToOrcasite
{
    class Program
    {
        static async Task Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("""
Usage: PostBackfillToOrcasite <ISO8601 timestamp>

where <ISO8601 timestamp> represents a minimum timestamp
after which any detections are to be posted to Orcasite.

The following environment variables must be set:
 * aifororcasmetadatastore_DOCUMENTDB
     Must be set to the OrcaHello cosmosdb connection string.
 * ORCASITE_APIKEY
     Must be set to the API key to allow Orcasite posts.
 * ORCASITE_HOSTNAME
     Should be set to the Orcasite hostname.
     Defaults to beta.orcasound.net if not present.

Example:
    PostBackfillToOrcasite 2025-07-30T00:00:00Z

    This example will copy all detections since July 30, 2025
    to Orcasite.
""");
                return;
            }
            if (!DateTime.TryParse(args[0], out DateTime startTime))
            {
                Console.WriteLine("Invalid timestamp format. Use ISO 8601 (e.g., 2023-01-01T00:00:00Z)");
                return;
            }
            string isoTimestamp = startTime.ToString("o"); // ISO 8601 format

            // Set up Orcasite client.
            var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });
            var helperLogger = loggerFactory.CreateLogger<OrcasiteHelper>();
            var httpClient = new HttpClient();
            var orcasiteHelper = new OrcasiteHelper(helperLogger, httpClient);
            await orcasiteHelper.InitializeAsync();
            var functionLogger = loggerFactory.CreateLogger<PostToOrcasite>();
            var postToOrcasite = new PostToOrcasite(orcasiteHelper, functionLogger);

            // Set up Cosmos DB client.
            string connectionString = Environment.GetEnvironmentVariable("aifororcasmetadatastore_DOCUMENTDB")
                ?? throw new InvalidOperationException("aifororcasmetadatastore_DOCUMENTDB not set");
            string databaseName = "predictions";
            string containerName = "metadata";
            var client = new CosmosClient(connectionString);
            var container = client.GetContainer(databaseName, containerName);

            // Start reading OrcaHello detections.
            Console.WriteLine($"Querying for detections with timestamp after {isoTimestamp}");
            var query = container.GetItemQueryIterator<JObject>(
                new QueryDefinition("SELECT * FROM c WHERE c.timestamp > @startTime")
                    .WithParameter("@startTime", isoTimestamp));

            int count = 0;
            int successes = 0;
            while (query.HasMoreResults)
            {
                foreach (JObject item in await query.ReadNextAsync())
                {
                    count++;

                    // Convert JObject to JsonElement.
                    string jsonString = item.ToString(Newtonsoft.Json.Formatting.None);
                    JsonElement element = JsonDocument.Parse(jsonString).RootElement;

                    // Try posting the detection to Orcasite.
                    // This will fail if it is already present there.
                    var documents = new List<JsonElement> { element };

                    // Currently all data in the OrcaHello database has incorrect timestamps.
                    // We correct for that here.
                    // TODO(issue #219): Update this workaround when the data is fixed.
                    IReadOnlyList<JsonElement> correctedDocuments = await orcasiteHelper.FixTimestampsAsync(documents);

                    bool ok = await postToOrcasite.ProcessDocumentsAsync(correctedDocuments);
                    if (ok)
                    {
                        successes++;
                    }

                    // Output the result.
                    string timestamp = element.GetProperty("timestamp").GetDateTime().ToString();
                    Console.WriteLine($"#{count}: Posting {timestamp} => {ok}");
                }
            }

            Console.WriteLine($"Total items read: {count}");
            Console.WriteLine($"Total items posted: {successes}");
        }
    }
}
