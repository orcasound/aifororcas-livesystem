namespace AIForOrcas.Client.Web.Models.Settings;

public class AppSettings
{
    public const string Section = "AppSettings";
    public string APIUrl { get; set; }
    public string[] Locations { get; set; }
    public AzureAd AzureAd { get; set; }
}
