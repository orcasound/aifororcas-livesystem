using Microsoft.Azure.Cosmos;
using Microsoft.Azure.Documents;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Tests.Common;

namespace NotificationSystem.Tests.Integration
{
    public class PostToOrcasiteIntegrationTests : IDisposable
    {
        private readonly IHost _host;
        private readonly IServiceProvider _serviceProvider;
        private readonly string _solutionDirectory;

        /// <summary>
        /// Get the contents of a TestData file as a string.
        /// </summary>
        /// <param name="filename">Name of file to load</param>
        /// <returns>String contents</returns>
        private string GetStringFromFile(string filename)
        {
            return File.ReadAllText(Path.Combine(_solutionDirectory, "TestData", filename));
        }

        public PostToOrcasiteIntegrationTests()
        {
            _solutionDirectory = OrcasiteTestHelper.FindSolutionDirectory() ?? throw new Exception("Could not find solution directory");
            _host = CreateHostBuilder().Build();
            _serviceProvider = _host.Services;
        }

        /// <summary>
        /// Create a test host with proper dependency injection configuration.
        /// </summary>
        /// <returns>Host builder configured for integration testing</returns>
        private IHostBuilder CreateHostBuilder()
        {
            return Host.CreateDefaultBuilder()
                .ConfigureAppConfiguration((context, config) =>
                {
                    // Add default environment variable values.
                    config.AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        ["aifororcasmetadatastore_DOCUMENTDB"] = Environment.GetEnvironmentVariable("aifororcasmetadatastore_DOCUMENTDB") ?? "UseDevelopmentStorage=true",
                        ["ORCASITE_HOSTNAME"] = Environment.GetEnvironmentVariable("ORCASITE_HOSTNAME") ?? "beta.orcasound.net",
                        ["ORCASITE_APIKEY"] = Environment.GetEnvironmentVariable("ORCASITE_APIKEY") ?? "test-key"
                    });

                    // Load configuration from local.settings.json.
                    string? solutionDir = OrcasiteTestHelper.FindSolutionDirectory();
                    string? functionAppDir = solutionDir != null
                        ? Path.Combine(solutionDir, "NotificationSystem")
                        : null;

                    if (functionAppDir != null)
                    {
                        string settingsPath = Path.Combine(functionAppDir, "local.settings.json");
                        if (File.Exists(settingsPath))
                        {
                            var rawSettings = new ConfigurationBuilder()
                                .AddJsonFile(settingsPath, optional: false, reloadOnChange: false)
                                .Build();

                            // Flatten the "Values" section into the main config
                            var valuesSection = rawSettings.GetSection("Values").GetChildren();
                            var flattened = valuesSection.ToDictionary(x => x.Key, x => x.Value);

                            config.AddInMemoryCollection(flattened);
                        }
                    }
                })
                .ConfigureServices((context, services) =>
                {
                    // Register services that would normally be injected in Azure Functions.
                    services.AddSingleton<ILoggerFactory, LoggerFactory>();
                    services.AddSingleton(typeof(ILogger<>), typeof(Logger<>));
                    services.AddTransient<ILogger>(provider => provider.GetService<ILogger<PostToOrcasiteIntegrationTests>>()!);
                    services.AddSingleton<OrcasiteHelper>(provider =>
                    {
                        var logger = provider.GetRequiredService<ILogger<PostToOrcasiteIntegrationTests>>();
                        return OrcasiteTestHelper.GetMockOrcasiteHelper(logger);
                    });
                    services.AddTransient<PostToOrcasite>(provider =>
                    {
                        var orcasiteHelper = provider.GetRequiredService<OrcasiteHelper>();
                        return new PostToOrcasite(orcasiteHelper);
                    });
                    
                    // Register Cosmos DB services if connection string is available.
                    var connectionString = context.Configuration["aifororcasmetadatastore_DOCUMENTDB"];
                    if (!string.IsNullOrEmpty(connectionString) && connectionString != "UseDevelopmentStorage=true")
                    {
                        services.AddSingleton<CosmosClient>(provider =>
                        {
                            return new CosmosClient(connectionString);
                        });
                    }
                })
                .ConfigureLogging(logging =>
                {
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Information);
                });
        }

        /// <summary>
        /// Integration test for UpdateCosmosDb following DI pattern.
        /// This test uses the actual services configured through dependency injection,
        /// similar to how they would be used in the Azure Functions runtime.
        /// </summary>
        [Fact]
        public async Task UpdateCosmosDb_WithDependencyInjection()
        {
            // Arrange - Get services from DI container.
            var logger = _serviceProvider.GetRequiredService<ILogger>();
            var postToOrcasite = _serviceProvider.GetRequiredService<PostToOrcasite>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();

            // Check if we have a real Cosmos DB connection.
            var connectionString = configuration["aifororcasmetadatastore_DOCUMENTDB"];
            if (string.IsNullOrEmpty(connectionString) || connectionString == "UseDevelopmentStorage=true")
            {
                // Skip test if no real Cosmos DB connection is available.
                logger.LogWarning("Skipping integration test - no Cosmos DB connection string configured");
                return;
            }

            // Get Cosmos DB client from DI container.
            var cosmosClient = _serviceProvider.GetService<CosmosClient>();
            if (cosmosClient == null)
            {
                logger.LogWarning("Skipping integration test - Cosmos DB client not available");
                return;
            }

            // Create test data.
            List<Document> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Act - Test the actual function logic using DI services.
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents, logger);
            Assert.True(ok, "PostToOrcasite failed");

            // Assert - Verify the function executed without throwing.
            // In a real integration test, you would verify the document was processed correctly.
            // by checking the database state or external API calls.
            logger.LogInformation($"Successfully processed document");
        }

        /// <summary>
        /// Integration test that calls the actual PostToOrcasite Azure Function entry point.
        /// This test invokes the [FunctionName("PostToOrcasite")] method directly.
        /// </summary>
        [Fact]
        public async Task PostToOrcasite_AzureFunctionEntryPoint_WithDependencyInjection()
        {
            // Arrange - Get services from DI container.
            var logger = _serviceProvider.GetRequiredService<ILogger>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();
            
            // Check if we have a real Cosmos DB connection.
            var connectionString = configuration["aifororcasmetadatastore_DOCUMENTDB"];
            if (string.IsNullOrEmpty(connectionString) || connectionString == "UseDevelopmentStorage=true")
            {
                // Skip test if no real Cosmos DB connection is available.
                logger.LogWarning("Skipping PostToOrcasite Azure Function integration test - no Cosmos DB connection string configured");
                return;
            }

            List<Document> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Act - Call the actual Azure Function entry point.
            // This tests the [FunctionName("PostToOrcasite")] method directly.
            var postToOrcasite = _serviceProvider.GetRequiredService<PostToOrcasite>();
            int oldRunCount = postToOrcasite.SuccessfulRuns;
            await postToOrcasite.Run(documents, logger);

            // Assert - Verify the function executed without throwing.
            // In a real integration test, you would verify the document was processed correctly
            // by checking the external API calls or side effects.
            Assert.NotEqual(oldRunCount, postToOrcasite.SuccessfulRuns);
            logger.LogInformation($"Successfully executed PostToOrcasite Azure Function");
        }

        /// <summary>
        /// Test the Cosmos DB container connection using DI services.
        /// </summary>
        [Fact]
        public async Task CosmosDbContainer_ConnectionTest_WithDependencyInjection()
        {
            // Arrange.
            var logger = _serviceProvider.GetRequiredService<ILogger>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();

            var connectionString = configuration["aifororcasmetadatastore_DOCUMENTDB"];
            if (string.IsNullOrEmpty(connectionString) || connectionString == "UseDevelopmentStorage=true")
            {
                logger.LogWarning("Skipping Cosmos DB connection test - no connection string configured");
                return;
            }

            // Act - Test Cosmos DB connection using DI.
            var cosmosClient = _serviceProvider.GetService<CosmosClient>();
            if (cosmosClient == null)
            {
                logger.LogWarning("Skipping Cosmos DB connection test - client not available");
                return;
            }

            // Try to get the database and container to verify connection.
            var database = cosmosClient.GetDatabase("predictions");
            var container = database.GetContainer("metadata");

            // Perform a simple read operation to test the connection.
            var query = new QueryDefinition("SELECT TOP 1 * FROM c");
            using var iterator = container.GetItemQueryIterator<dynamic>(query);

            if (iterator.HasMoreResults)
            {
                var response = await iterator.ReadNextAsync();
                logger.LogInformation($"Successfully connected to Cosmos DB. Response status: {response.StatusCode}");
            }

            // Assert - If we reach here without exception, the connection is working.
            Assert.True(true, "Cosmos DB connection test passed");
        }

        public void Dispose()
        {
            _host?.Dispose();
        }
    }
}
