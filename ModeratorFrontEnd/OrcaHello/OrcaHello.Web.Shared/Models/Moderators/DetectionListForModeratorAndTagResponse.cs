namespace OrcaHello.Web.Shared.Models.Moderators
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A paginated list of Detections for a given timeframe, tag and moderator.")]
    public class DetectionListForModeratorAndTagResponse : DetectionListForTagResponse
    {
        [SwaggerSchema("The unique names of the moderator.")]
        public string Moderator { get; set; }
    }
}
