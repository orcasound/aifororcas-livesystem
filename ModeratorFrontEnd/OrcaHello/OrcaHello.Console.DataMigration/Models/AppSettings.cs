namespace OrcaHello.Console.DataMigration.Models
{
    public class AppSettings
    {
        public const string CosmosDb = "OnlineCosmosDbConnectionString";
        public const string LocalDb = "LocalCosmosDbEmulatorConnectionString";
        public const string SourceDbName = "SourceDbName";
        public const string SourceContainerName = "SourceContainerName";
        public const string SourcePartitionKey = "SourcePartitionKey";
        public const string TargetDbName = "TargetDbName";
        public const string TargetContainerName = "TargetContainerName";
        public const string TargetPartitionKey = "TargetPartitionKey";
    }
}
