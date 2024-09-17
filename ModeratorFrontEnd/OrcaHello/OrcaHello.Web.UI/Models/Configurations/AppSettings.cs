namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class AppSettings
    {
        public const string Section = "AppSettings";

        public List<string> HydrophoneLocationNames { get; set; } = new();
        public string APIUrl { get; set; } = string.Empty;
        public AzureAd AzureAd { get; set; } = new();
        public string EpochDateString { get; set; } = string.Empty;

        public DateTime EpochDate => DateTime.Parse(EpochDateString);
    }
}
