namespace OrcaHello.Web.Shared.Models.Moderators
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A list of tags for the given timeframe and moderator")]
    public class TagListForModeratorResponse : TagListForTimeframeResponse
    {
        [SwaggerSchema("The name of the moderator.")]
        public string Moderator { get; set; }
    }
}
