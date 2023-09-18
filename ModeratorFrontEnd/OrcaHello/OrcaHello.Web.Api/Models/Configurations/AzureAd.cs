namespace OrcaHello.Web.Api.Models.Configurations
{
    [ExcludeFromCodeCoverage]
    public class AzureAd
    {
        public string Instance { get; set; }
        public string Domain { get; set; }
        public string TenantId { get; set; }
        public string ClientId { get; set; }
        public string Scopes { get; set; }
        public string CallbackPath { get; set; }
        public string ModeratorGroupId { get; set; }
    }
}
