namespace OrcaHello.Web.Shared.Models.Tags
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The results of a Tag removal request.")]
    public class TagRemovalResponse
    {
        [SwaggerSchema("The Tag to remove.")]
        public string Tag { get; set; }
        [SwaggerSchema("The total number of Detections with the Tag.")]
        public int TotalMatching { get; set; }

        [SwaggerSchema("The total number of Detections where the Tag was successfully removed.")]
        public int TotalRemoved { get; set; }
    }
}
