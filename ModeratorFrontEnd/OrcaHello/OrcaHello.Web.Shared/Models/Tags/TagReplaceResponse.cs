namespace OrcaHello.Web.Shared.Models.Tags
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The results of a Tag replace request.")]
    public class TagReplaceResponse
    {
        [SwaggerSchema("The Tag to replace.")]
        public string OldTag { get; set; }
        [SwaggerSchema("The Tag to replace it with.")]
        public string NewTag { get; set; }

        [SwaggerSchema("The total number of Detections with the Tag.")]
        public int TotalMatching { get; set; }

        [SwaggerSchema("The total number of Detections where the Tag was successfully replaced.")]
        public int TotalReplaced { get; set; }
    }
}
