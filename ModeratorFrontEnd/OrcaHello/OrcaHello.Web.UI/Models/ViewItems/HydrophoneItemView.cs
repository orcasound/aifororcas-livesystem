namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class HydrophoneItemView
    {
        public string Id { get; set; } = null!;
        public string Name { get; set; } = null!;
        public double Longitude { get; set; }
        public double Latitude { get; set; }
        public string ImageUrl { get; set; } = null!;
        public string IntroHtml { get; set; } = null!;
    }
}
