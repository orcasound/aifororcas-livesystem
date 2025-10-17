using ComposableAsync;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class PostToOrcasite
    {
        private readonly OrcasiteHelper _helper;
        private readonly ILogger _logger;
        private readonly IConfiguration _configuration;
        const int SendRate = 14; // Max 14 posts per second.

        public PostToOrcasite(OrcasiteHelper helper, ILogger<PostToOrcasite> logger, IConfiguration configuration)
        {
            _helper = helper;
            _logger = logger;
            _configuration = configuration;
        }

        public async Task<bool> ProcessDocumentsAsync(
            IReadOnlyList<JsonElement> input)
        {
            if (input == null || input.Count == 0)
            {
                _logger.LogInformation("No updated records");
                return true;
            }

            await _helper.InitializeAsync(_configuration);

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            int remaining = input.Count;

            foreach (var document in input)
            {
                bool ok = await _helper.PostDetectionAsync(document.ToString());
                if (!ok)
                {
                    return false;
                }
                remaining--;
                if (remaining > 0)
                {
                    await timeConstraint;
                }
            }
            return true;
        }

        int _successfulRuns = 0;
        public int SuccessfulRuns => _successfulRuns;

        [Function("PostToOrcasite")]
        public async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                containerName: "metadata",
                Connection = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseContainerName = "leases",
                LeaseContainerPrefix = "orcasite",
                CreateLeaseContainerIfNotExists = true)] IReadOnlyList<JsonElement> input)
        {
            await _helper.InitializeAsync(_configuration);

            // Data in the OrcaHello database from before the current epoch has
            // incorrect timestamps, so correct for that here.
            IReadOnlyList<JsonElement> correctedInput = await _helper.FixTimestampsAsync(input);

            bool ok = await ProcessDocumentsAsync(correctedInput);
            if (ok)
            {
                Interlocked.Increment(ref _successfulRuns);
            }
        }
    }
}
