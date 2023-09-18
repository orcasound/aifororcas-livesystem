namespace OrcaHello.Web.Shared.Models.Detections
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A request for updating a detection with moderator-related information.")]
    public class ModerateDetectionRequest
    {
        [SwaggerSchema("The detection id.")]
        public string Id { get; set; }

        [SwaggerSchema("The detection's new state.")]
        public string State { get; set; }

        [SwaggerSchema("The name of the moderator.")]
        public string Moderator { get; set; }

        [SwaggerSchema("The datetime the detection was moderated.")]
        public DateTime DateModerated { get; set; }

        [SwaggerSchema("Comments the moderator made about the detection.")]
        public string Comments { get; set; }

        [SwaggerSchema("The list of tags the moderator added to the detection.")]
        public List<string> Tags { get; set; }
    }
}
