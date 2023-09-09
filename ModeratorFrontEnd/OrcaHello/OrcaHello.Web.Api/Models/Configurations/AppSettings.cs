namespace OrcaHello.Web.Api.Models.Configurations
{
    [ExcludeFromCodeCoverage]
    public class AppSettings
    {
        public const string Section = "AppSettings";
        public string CosmosConnectionString { get; set; }
        public string DetectionsDatabaseName { get; set; }
        public string MetadataContainerName { get; set; }
        public string AllowedOrigin { get; set; }
        public AzureAd AzureAd { get; set; }
        public virtual List<HydrophoneLocation> HydrophoneLocations { get; set; }
    }
}
