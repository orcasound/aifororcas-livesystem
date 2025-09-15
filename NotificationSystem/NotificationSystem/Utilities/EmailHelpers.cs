using Amazon.SimpleEmail.Model;
using Azure.Data.Tables;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using NotificationSystem.Models;
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Mail;
using System.Threading.Tasks;

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

        public static async Task<HttpResponseData> UpdateEmailListAsync<T>(
            HttpRequestData req,
            TableClient tableClient,
            ILogger log) where T : EmailEntity, ITableEntity, new()
        {
            string requestBody = await req.ReadAsStringAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            string email = data?.email;

            if (string.IsNullOrWhiteSpace(email) || !IsValidEmail(email))
            {
                // non-valid email address
                log.LogInformation($"Received non-valid email address: {email}");
                var badResponse = req.CreateResponse(HttpStatusCode.BadRequest);
                await badResponse.WriteStringAsync($"Received non-valid email address: {email}\n");
                return badResponse;
            }

            try
            {
                var emailEntity = (T)Activator.CreateInstance(typeof(T), email);

                if (req.Method.Equals("POST", StringComparison.OrdinalIgnoreCase))
                {
                    await tableClient.UpsertEntityAsync(emailEntity);
                }
                else if (req.Method.Equals("DELETE", StringComparison.OrdinalIgnoreCase))
                {
                    await tableClient.DeleteEntityAsync(emailEntity.PartitionKey, emailEntity.RowKey);
                }
                else
                {
                    var methodNotAllowed = req.CreateResponse(HttpStatusCode.MethodNotAllowed);
                    await methodNotAllowed.WriteStringAsync("Only POST and DELETE are supported.");
                    return methodNotAllowed;
                }
            }
            catch (Exception e)
            {
                log.LogError($"Unable to {req.Method} email: {email}. Exception: {e.Message}");
                var errorResponse = req.CreateResponse(HttpStatusCode.BadRequest);
                await errorResponse.WriteStringAsync($"Unable to {req.Method} email: {email}. Exception: {e.Message}\n");
                return errorResponse;
            }

            log.LogInformation($"Successfully {req.Method} email: {email}");
            var successResponse = req.CreateResponse(HttpStatusCode.OK);
            await successResponse.WriteStringAsync($"Successfully {req.Method} email: {email}\n");
            return successResponse;
        }

        public static async Task<IEnumerable<T>> GetEmailEntitiesAsync<T>(
    TableClient tableClient,
    string partitionKey) where T : class, ITableEntity, new()
        {
            var results = new List<T>();
            await foreach (var entity in tableClient.QueryAsync<T>(e => e.PartitionKey == partitionKey))
            {
                results.Add(entity);
            }
            return results;
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