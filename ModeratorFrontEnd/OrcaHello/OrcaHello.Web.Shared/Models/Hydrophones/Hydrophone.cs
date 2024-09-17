namespace OrcaHello.Web.Shared.Models.Hydrophones
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("Hydrophone location information.")]
    public class Hydrophone
    {
        [SwaggerSchema("The hydrophone's id.")]
        public string Id { get; set; }

        [SwaggerSchema("The hydrophone's location name.")]
        public string Name { get; set; }

        [SwaggerSchema("The hydrophone's location longitude.")]
        public double Longitude { get; set; }

        [SwaggerSchema("The hydrophone's location latitude.")]
        public double Latitude { get; set; }

        [SwaggerSchema("Link to an image showing the hydrophone location.")]
        public string ImageUrl { get; set; }

        [SwaggerSchema("HTML-formatted description of the hydrophone location.")]
        public string IntroHtml { get; set; }

        public override string ToString() => Name;
    }
}
