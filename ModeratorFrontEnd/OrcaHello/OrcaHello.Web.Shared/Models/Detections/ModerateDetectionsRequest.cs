namespace OrcaHello.Web.Shared.Models.Detections
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A request for updating one or more detections with moderator-related information.")]
    public class ModerateDetectionsRequest
    {
        [SwaggerSchema("The detection ids to be moderated.")]
        public List<string> Ids { get; set; }

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
