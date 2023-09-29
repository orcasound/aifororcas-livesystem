namespace OrcaHello.Web.Shared.Models.Detections
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A response indicating the success or failure of a ModerateDetectionsRequest")]
    public class ModerateDetectionsResponse
    {
        public List<string> IdsToUpdate { get; set; } = new();
        public List<string> IdsNotFound { get; set; } = new();
        public List<string> IdsSuccessfullyUpdated { get; set; } = new();
        public List<string> IdsUnsuccessfullyUpdated { get; set; } = new();
    }
}
