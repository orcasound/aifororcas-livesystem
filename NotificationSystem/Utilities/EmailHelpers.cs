using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Mail;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Cosmos.Table;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using NotificationSystem.Models;
using Amazon.SimpleEmail.Model;

namespace NotificationSystem.Utilities
{
    public class EmailHelpers
    {
        public static bool IsValidEmail(string emailaddress)
        {
            try
            {
                MailAddress m = new MailAddress(emailaddress);

                return true;
            }
            catch (FormatException)
            {
                return false;
            }
        }

        public static async Task<IActionResult> UpdateEmailList<T>(HttpRequest req, CloudTable cloudTable, ILogger log)
            where T : EmailEntity
        {
            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            string email = data?.email;
            string responseMessage;

            if (string.IsNullOrWhiteSpace(email) || !IsValidEmail(email))
            {
                // non-valid email address
                log.LogInformation($"Received non-valid email address: {email}");
                responseMessage = $"Received non-valid email address: {email}\n";
                return new BadRequestObjectResult(responseMessage);
            }

            try
            {
                var emailEntity = (T)Activator.CreateInstance(typeof(T), email);
                var operation = req.Method.Equals("post", StringComparison.OrdinalIgnoreCase) ?
                    TableOperation.Insert(emailEntity) :
                    TableOperation.Delete(emailEntity);
                await cloudTable.ExecuteAsync(operation);
            }
            catch (StorageException e)
            {
                log.LogError($"Unable to {req.Method} email: {email}. Exception: {e.Message}");
                responseMessage = $"Unable to {req.Method} email: {email}. Exception: {e.Message}\n";
                return new BadRequestObjectResult(responseMessage);
            }

            log.LogInformation($"Successfully {req.Method} email: {email}");
            responseMessage = $"Successfully {req.Method} email: {email}\n";
            return new OkObjectResult(responseMessage);
        }

        public static IEnumerable<EmailEntity> GetEmailEntities(CloudTable cloudTable, string type)
        {
            var query = new TableQuery<ModeratorEmailEntity>()
                .Where(TableQuery.GenerateFilterCondition("PartitionKey", QueryComparisons.Equal, type));
            return cloudTable.ExecuteQuery(query);
        }

		public static SendEmailRequest CreateEmail(string from, string to, string subject, string body)
		{
            var email = new SendEmailRequest();
            email.Source = from;
            email.Destination = new Destination(new List<string> { to });
            //Create message and attach to email request.
			Message message = new Message();
			message.Subject = new Content(subject);
			message.Body = new Body();
			message.Body.Html = new Content
			{
				Charset = "UTF-8",
				Data = body
			};
			email.Message = message;
			return email;
		}
	}
}