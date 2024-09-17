namespace OrcaHello.Web.Shared.Models.Detections
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A hydrophone sampling that might contain whale sounds.")]
    public class Detection
    {
        [SwaggerSchema("The detection's generated unique Id.")]
        public string Id { get; set; }

        [SwaggerSchema("URI of the detection's audio file (.wav) in blob storage.")]
        public string AudioUri { get; set; }

        [SwaggerSchema("URI of the detection's image file (.png) in blob storage.")]
        public string SpectrogramUri { get; set; }

        [SwaggerSchema("The review state of the detection (Unreviewed, Positive, Negative, or Unknown).")]
        public string State { get; set; }

        [SwaggerSchema("The name of the hydrophone's location.")]
        public string LocationName { get; set; }

        [SwaggerSchema("Calculated average confidence that the detection contains a whale sound.")]
        public decimal Confidence { get; set; }

        [SwaggerSchema("Location geodata.")]
        public Location Location { get; set; }

        [SwaggerSchema("Date and time of when the detection occurred.")]
        public DateTime Timestamp { get; set; }

        [SwaggerSchema("List of sections within the detection that might contain whale sounds.")]
        public List<Annotation> Annotations { get; set; } = new List<Annotation>();

        [SwaggerSchema("List of tags entered by the human moderator during review.")]
        public List<string> Tags { get; set; } = new List<string>();

        [SwaggerSchema("A special interest label entered by the human moderator during review.")]
        public string InterestLabel { get; set; }

        [SwaggerSchema("Any text comments entered by the human moderator during review.")]
        public string Comments { get; set; }

        [SwaggerSchema("Identity of the human moderator (User Principal Name for AzureAD) performing the review.")]
        public string Moderator { get; set; }

        [SwaggerSchema("Date and time of when the detection was reviewed by the human moderator.")]
        public DateTime? Moderated { get; set; }
    }

    [ExcludeFromCodeCoverage]
    [SwaggerSchema("Geographical location of the hydrophone that collected the detection.")]
    public class Location
    {
        [SwaggerSchema("Name of the hydrophone location.")]
        public string Name { get; set; }
        [SwaggerSchema("Longitude of the hydrophone's location.")]
        public double Longitude { get; set; }
        [SwaggerSchema("Latitude of the hydrophone's location.")]
        public double Latitude { get; set; }
    }

    [ExcludeFromCodeCoverage]
    [SwaggerSchema("Section within the detection that might contain whale sounds.")]
    public class Annotation
    {
        [SwaggerSchema("Unique identifier (within the detection) of the annotation.")]
        public int Id { get; set; }
        [SwaggerSchema("Start time (within the detection) of the annotation as measured in seconds.")]
        public decimal StartTime { get; set; }
        [SwaggerSchema("End time (within the detection) of the annotation as measured in seconds.")]
        public decimal EndTime { get; set; }
        [SwaggerSchema("Calculated confidence that the annotation contains a whale sound.")]
        public decimal Confidence { get; set; }
    }
}
