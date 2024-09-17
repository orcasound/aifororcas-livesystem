namespace OrcaHello.Web.Shared.Models.Tags
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("Replace one tag with another")]

    public class ReplaceTagRequest
    {
        [SwaggerSchema("The old tag to be replaced")]
        public string OldTag { get; set; } = null!;
        [SwaggerSchema("The new tag to replace it with")]
        public string NewTag { get; set; } = null!;
    }
}
