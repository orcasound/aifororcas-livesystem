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

        [SwaggerSchema("The hydrophone's location logitude.")]
        public double Longitude { get; set; }

        [SwaggerSchema("The hydrophone's location latitude.")]
        public double Latitude { get; set; }
    }
}
