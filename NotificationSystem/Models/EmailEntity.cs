using Microsoft.Azure.Cosmos.Table;

namespace NotificationSystem.Models
{
    public abstract class EmailEntity : TableEntity
    {
        public EmailEntity(string email, string type)
        {
            Email = email;
            PartitionKey = type;
            RowKey = email;
            ETag = "*";
        }
        public string Email { get; set; }
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