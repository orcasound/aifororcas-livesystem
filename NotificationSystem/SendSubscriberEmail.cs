using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos.Table;
using Microsoft.Azure.Storage.Queue;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NotificationSystem.Template;
using NotificationSystem.Utilities;
using SendGrid.Helpers.Mail;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public static class SendSubscriberEmail
    {
        [FunctionName("SendSubscriberEmail")]
        // TODO: change timer to once per hour (0 0 * * * *)
        public static async Task Run(
            [TimerTrigger("0 */1 * * * *")] TimerInfo myTimer,
            [Queue("srkwfound")] CloudQueue cloudQueue,
            [Table("EmailList")] CloudTable cloudTable,
            [SendGrid(ApiKey = "SendGridKey")] IAsyncCollector<SendGridMessage> messageCollector,
            ILogger log)
        {
            log.LogInformation("Checking if there are items in queue");
            await cloudQueue.FetchAttributesAsync();

            if (cloudQueue.ApproximateMessageCount == 0)
            {
                log.LogInformation("No items in queue");
                return;
            }

            log.LogInformation("Creating email message");
            var body = await CreateBody(cloudQueue);

            log.LogInformation("Retrieving email list and sending notifications");
            foreach (var emailEntity in EmailHelpers.GetEmailEntities(cloudTable, "Subscriber"))
            {
                var email = EmailHelpers.CreateEmail(Environment.GetEnvironmentVariable("SenderEmail"),
                    emailEntity.Email, "Notification: Orca detected!", body);
                await messageCollector.AddAsync(email);
            }
        }

        // TODO: make emails more pretty
        public static async Task<string> CreateBody(CloudQueue cloudQueue)
        {
            var bodyBuilder = new StringBuilder("<h1>Confirmed SRKW detections:</h1>\n<ul>");
            CloudQueueMessage message;
            List<JObject> messagesJson = new List<JObject>();

            while (true)
            {
                message = await cloudQueue.GetMessageAsync();
                if (message == null)
                    break;

                messagesJson.Add(JsonConvert.DeserializeObject<JObject>(message.AsString));
                await cloudQueue.DeleteMessageAsync(message);
            }

            return EmailTemplate.GetSubscriberEmailBody(messagesJson);
        }
    }
}
