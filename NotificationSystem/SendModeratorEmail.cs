using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos.Table;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using NotificationSystem.Template;
using NotificationSystem.Utilities;
using SendGrid.Helpers.Mail;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public static class SendModeratorEmail
    {
        [FunctionName("SendModeratorEmail")]
        public static async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                collectionName: "metadata",
                ConnectionStringSetting = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseCollectionName = "leases",
                LeaseCollectionPrefix = "moderator",
                CreateLeaseCollectionIfNotExists = true)]IReadOnlyList<Document> input,
            [Table("EmailList")] CloudTable cloudTable,
            [SendGrid(ApiKey = "SendGridKey")] IAsyncCollector<SendGridMessage> messageCollector,
            ILogger log)
        {
            if (input == null || input.Count == 0)
            {
                log.LogInformation("No updated records");
                return;
            }

            var newDocumentCreated = false;
            DateTime? documentTimeStamp = null;
            string location = null;

            foreach (var document in input)
            {
                if (!document.GetPropertyValue<bool>("reviewed"))
                {
                    newDocumentCreated = true;
                    documentTimeStamp = document.GetPropertyValue<DateTime>("timestamp");
                    location = document.GetPropertyValue<string>("location.name");
                    break;
                }
            }

            if (!newDocumentCreated)
            {
                log.LogInformation("No unreviewed records");
                return;
            }

            // TODO: make better email
            string body = EmailTemplate.GetModeratorEmailBody(documentTimeStamp, location);

            log.LogInformation("Retrieving email list and sending notifications");
            foreach (var emailEntity in EmailHelpers.GetEmailEntities(cloudTable, "Moderator"))
            {
                string emailSubject = string.Format("OrcaHello Candidate at location {0}", location);
                var email = EmailHelpers.CreateEmail(Environment.GetEnvironmentVariable("SenderEmail"),
                    emailEntity.Email, emailSubject, body);
                await messageCollector.AddAsync(email);
            }
        }
    }
}
