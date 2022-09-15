namespace AIForOrcas.Client.Web.Models.Settings;

public class AzureAd
{
    public string Instance { get; set; }
    public string Domain { get; set; }
    public string TenantId { get; set; }
    public string ClientId { get; set; }
    public string DefaultScope { get; set; }
    public AzureAdGroups AzureAdGroups { get; set; }
}

public class AzureAdGroups
{
    public string ModeratorGroupId { get; set; }
}
