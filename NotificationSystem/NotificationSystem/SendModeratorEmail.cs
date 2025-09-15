using Amazon;
using Amazon.SimpleEmail;
using Azure.Data.Tables;
using ComposableAsync;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Template;
using NotificationSystem.Utilities;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class SendModeratorEmail
    {
        private readonly ILogger _logger;
        const int SendRate = 14;

        public SendModeratorEmail(ILogger<SendModeratorEmail> logger)
        {
            _logger = logger;
        }

        [Function("SendModeratorEmail")]
        public async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                containerName: "metadata",
                Connection = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseContainerName = "leases",
                LeaseContainerPrefix = "moderator",
                CreateLeaseContainerIfNotExists = true)] IReadOnlyList<JsonElement> input,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient)
        {
            if (input == null || input.Count == 0)
            {
                _logger.LogInformation("No updated records");
                return;
            }

            var newDocumentCreated = false;
            DateTime? documentTimeStamp = null;
            string location = null;

            foreach (var document in input)
            {
                // Check whether the "reviewed" property exists.
                JsonElement reviewed = document.GetProperty("reviewed");
                if (reviewed.ValueKind != JsonValueKind.True)
                {
                    newDocumentCreated = true;
                    documentTimeStamp = document.GetProperty("timestamp").GetDateTime();
                    location = document.GetProperty("location").GetProperty("name").GetString();
                }
            }

            if (!newDocumentCreated)
            {
                _logger.LogInformation("No unreviewed records");
                return;
            }

            string body = EmailTemplate.GetModeratorEmailBody(documentTimeStamp, location);

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            var aws = new AmazonSimpleEmailServiceClient(RegionEndpoint.USWest2);
            _logger.LogInformation("Retrieving email list and sending notifications");
            foreach (var emailEntity in await EmailHelpers.GetEmailEntitiesAsync<ModeratorEmailEntity>(tableClient, "Moderator"))
            {
                await timeConstraint;
                string emailSubject = $"OrcaHello Candidate at location {(string.IsNullOrEmpty(location) ? "Unknown" : location)}";
                var email = EmailHelpers.CreateEmail(Environment.GetEnvironmentVariable("SenderEmail"),
                    emailEntity.Email, emailSubject, body);
                await aws.SendEmailAsync(email);
            }
        }
    }
}
