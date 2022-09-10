namespace AIForOrcas.Server.Models.Settings;

public class AppSettings
{
    public const string Section = "AppSettings";
    public string ProviderAccountEndpoint { get; set; }
    public string ProviderAccountKey { get; set; }
    public string DatabaseName { get; set; }
    public string AllowedOrigin { get; set; }
    public AzureAd AzureAd { get; set; }
}
