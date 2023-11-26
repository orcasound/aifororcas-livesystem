namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class AzureAd
    {
        public string Instance { get; set; } = string.Empty;
        public string Domain { get; set; } = string.Empty;
        public string TenantId { get; set; } = string.Empty;
        public string ClientId { get; set; } = string.Empty;
        public string DefaultScope { get; set; } = string.Empty;
        public AzureAdGroups AzureAdGroups { get; set; } = new();
    }

    [ExcludeFromCodeCoverage]
    public class AzureAdGroups
    {
        public string ModeratorGroupId { get; set; } = string.Empty;
    }
}
