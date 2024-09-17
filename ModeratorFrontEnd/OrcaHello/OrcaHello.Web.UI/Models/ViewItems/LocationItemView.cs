namespace OrcaHello.Web.UI.Models
{
    // This class represents a view model for a location item, which is used throughout the
    // various view services to display information about locations.
    [ExcludeFromCodeCoverage]
    public class LocationItemView
    {
        // Name of the location.
        public string Name { get; set; } = string.Empty;

        // Longitude coordinate of the location.
        public double Longitude { get; set; } = double.MinValue;

        // Latitude coordinate of the location.
        public double Latitude { get; set; } = double.MinValue;

        // Converts a Location object (from the API) into a LocationItemView using a delegate.

        public static Func<Location, LocationItemView> AsLocationItemView =>
            location => new LocationItemView
            {
                Name = location.Name,
                Longitude = location.Longitude,
                Latitude = location.Latitude
            };
    }
}
