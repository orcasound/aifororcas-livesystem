using Azure;
using Azure.Data.Tables;
using System;

namespace NotificationSystem.Models
{
    public abstract class EmailEntity : ITableEntity
    {
        protected EmailEntity() { }

        protected EmailEntity(string email, string type)
        {
            Email = email.ToLowerInvariant();
            PartitionKey = type;
            RowKey = email.ToLowerInvariant();
            ETag = ETag.All;
        }

        public string Email { get; set; }
        public string PartitionKey { get; set; }
        public string RowKey { get; set; }
        public DateTimeOffset? Timestamp { get; set; }
        public ETag ETag { get; set; }
    }

    public class SubscriberEmailEntity : EmailEntity
    {
        public SubscriberEmailEntity() : this("") { }
        public SubscriberEmailEntity(string email) : base(email, "Subscriber") { }
    }

    public class ModeratorEmailEntity : EmailEntity
    {
        public ModeratorEmailEntity() : this("") { }
        public ModeratorEmailEntity(string email) : base(email, "Moderator") { }
    }
}