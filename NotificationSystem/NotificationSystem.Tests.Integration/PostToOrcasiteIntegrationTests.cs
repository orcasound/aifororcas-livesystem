using Microsoft.Azure.Cosmos;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Tests.Common;
using System.Text.Json;

namespace NotificationSystem.Tests.Integration
{
    public class PostToOrcasiteIntegrationTests : IDisposable
    {
        private readonly IHost _host;
        private readonly IServiceProvider _serviceProvider;
        private readonly string _solutionDirectory;

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
                    string? functionAppDir = _solutionDirectory != null
                        ? Path.Combine(_solutionDirectory, "NotificationSystem")
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
                    services.AddSingleton<OrcasiteHelper>(provider =>
                    {
                        var logger = provider.GetRequiredService<ILogger<OrcasiteHelper>>();
                        return OrcasiteTestHelper.GetMockOrcasiteHelper(logger);
                    });
                    services.AddTransient<PostToOrcasite>(provider =>
                    {
                        var logger = provider.GetRequiredService<ILogger<PostToOrcasite>>();
                        var orcasiteHelper = provider.GetRequiredService<OrcasiteHelper>();
                        return new PostToOrcasite(orcasiteHelper, logger);
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
        [SkippableFact]
        public async Task UpdateCosmosDb_WithDependencyInjection()
        {
            // Arrange - Get services from DI container.
            var postToOrcasite = _serviceProvider.GetRequiredService<PostToOrcasite>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();

            // Skip test if no real Cosmos DB connection is available.
            var connectionString = configuration["aifororcasmetadatastore_DOCUMENTDB"];
            Skip.If(string.IsNullOrEmpty(connectionString) || connectionString == "UseDevelopmentStorage=true", "no Cosmos DB connection string configured");

            // Get Cosmos DB client from DI container.
            var cosmosClient = _serviceProvider.GetService<CosmosClient>();
            Skip.If(cosmosClient == null, "Cosmos DB client not available");

            // Create test data.
            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Act - Test the actual function logic using DI services.
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents);

            // Assert - Verify the function succeeded.
            Assert.True(ok, "PostToOrcasite failed");
        }

        /// <summary>
        /// Integration test that calls the actual PostToOrcasite Azure Function entry point.
        /// This test invokes the [FunctionName("PostToOrcasite")] method directly.
        /// </summary>
        [SkippableFact]
        public async Task PostToOrcasite_AzureFunctionEntryPoint_WithDependencyInjection()
        {
            // Arrange - Get services from DI container.
            var logger = _serviceProvider.GetRequiredService<ILogger<PostToOrcasiteIntegrationTests>>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();

            // Skip test if no real Cosmos DB connection is available.
            var connectionString = configuration["aifororcasmetadatastore_DOCUMENTDB"];
            Skip.If(string.IsNullOrEmpty(connectionString) || connectionString == "UseDevelopmentStorage=true", "no Cosmos DB connection string configured");

            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Act - Call the actual Azure Function entry point.
            // This tests the [FunctionName("PostToOrcasite")] method directly.
            var postToOrcasite = _serviceProvider.GetRequiredService<PostToOrcasite>();
            int oldRunCount = postToOrcasite.SuccessfulRuns;
            await postToOrcasite.Run(documents);

            // Assert - Verify the function executed without throwing.
            Assert.NotEqual(oldRunCount, postToOrcasite.SuccessfulRuns);
            logger.LogInformation($"Successfully executed PostToOrcasite Azure Function");
        }

        public void Dispose()
        {
            _host?.Dispose();
        }
    }
}
