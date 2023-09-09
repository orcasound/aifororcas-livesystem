namespace OrcaHello.Web.Shared.Models.Moderators
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A paginated list of Comments for a given timeframe and moderator.")]
    public class CommentListForModeratorResponse : CommentListResponse
    {
        [SwaggerSchema("The name of the moderator.")]
        public string Moderator { get; set; }
    }
}
