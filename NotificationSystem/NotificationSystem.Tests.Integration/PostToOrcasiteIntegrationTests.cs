using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Tests.Common;
using RichardSzalay.MockHttp;
using System.Text.Json;

namespace NotificationSystem.Tests.Integration
{
    public class PostToOrcasiteIntegrationTests : IDisposable
    {
        private readonly IHost _host;
        private readonly IServiceProvider _serviceProvider;
        private readonly string _solutionDirectory;
        private readonly MockHttpMessageHandler _mockHttp;
        private readonly MockedRequest _getFeedsRequest;
        private readonly MockedRequest _postDetectionRequest;

        public PostToOrcasiteIntegrationTests()
        {
            _solutionDirectory = OrcasiteTestHelper.FindSolutionDirectory() ?? throw new Exception("Could not find solution directory");
            _host = CreateHostBuilder().Build();
            _serviceProvider = _host.Services;
            
            // Get the mock HTTP handler and individual requests for verification purposes.
            var container = _serviceProvider.GetRequiredService<OrcasiteTestHelper.MockOrcasiteHelperContainer>();
            _mockHttp = container.MockHttp;
            _getFeedsRequest = container.GetFeedsRequest;
            _postDetectionRequest = container.PostDetectionRequest;
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
                    
                    // Register the container that holds both the helper and mock.
                    services.AddSingleton<OrcasiteTestHelper.MockOrcasiteHelperContainer>(provider =>
                    {
                        var logger = provider.GetRequiredService<ILogger<OrcasiteHelper>>();
                        return new OrcasiteTestHelper.MockOrcasiteHelperContainer(logger);
                    });
                    
                    // Register OrcasiteHelper by extracting from the container.
                    services.AddSingleton<OrcasiteHelper>(provider =>
                    {
                        var container = provider.GetRequiredService<OrcasiteTestHelper.MockOrcasiteHelperContainer>();
                        return container.Helper;
                    });
                    
                    // Register MockHttpMessageHandler by extracting from the container.
                    services.AddSingleton<MockHttpMessageHandler>(provider =>
                    {
                        var container = provider.GetRequiredService<OrcasiteTestHelper.MockOrcasiteHelperContainer>();
                        return container.MockHttp;
                    });
                    services.AddTransient<PostToOrcasite>(provider =>
                    {
                        var logger = provider.GetRequiredService<ILogger<PostToOrcasite>>();
                        var orcasiteHelper = provider.GetRequiredService<OrcasiteHelper>();
                        return new PostToOrcasite(orcasiteHelper, logger);
                    });
                })
                .ConfigureLogging(logging =>
                {
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Information);
                });
        }

        /// <summary>
        /// Integration test that calls the actual PostToOrcasite Azure Function entry point.
        /// This test invokes the [FunctionName("PostToOrcasite")] method directly.
        /// </summary>
        [Fact]
        public async Task PostToOrcasite_AzureFunctionEntryPoint_WithDependencyInjection()
        {
            // Arrange - Get services from DI container.
            var logger = _serviceProvider.GetRequiredService<ILogger<PostToOrcasiteIntegrationTests>>();
            var configuration = _serviceProvider.GetRequiredService<IConfiguration>();

            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Act - Call the actual Azure Function entry point.
            // This tests the [FunctionName("PostToOrcasite")] method directly.
            var postToOrcasite = _serviceProvider.GetRequiredService<PostToOrcasite>();
            int oldRunCount = postToOrcasite.SuccessfulRuns;
            await postToOrcasite.Run(documents);

            // Assert - Verify the function executed without throwing.
            Assert.NotEqual(oldRunCount, postToOrcasite.SuccessfulRuns);
            logger.LogInformation($"Successfully executed PostToOrcasite Azure Function");
            
            // Verify that the expected number of HTTP calls were made (1 GET feeds + 1 POST detection).
            Assert.Equal(1, _mockHttp.GetMatchCount(_getFeedsRequest));
            Assert.Equal(1, _mockHttp.GetMatchCount(_postDetectionRequest));
        }

        public void Dispose()
        {
            _host?.Dispose();
        }
    }
}
