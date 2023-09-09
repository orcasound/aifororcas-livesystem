namespace OrcaHello.Web.Shared.Models.Moderators
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A unique list of moderators.")]
    public class ModeratorListResponse
    {
        [SwaggerSchema("The unique names of the moderators.")]
        public List<string> Moderators { get; set; }
        [SwaggerSchema("The total number of moderators in the list.")]
        public int Count { get; set; }
    }
}
