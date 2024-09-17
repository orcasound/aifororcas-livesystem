namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class PaginatedCommentsByDateRequest : DateRequestBase
    {
        public int Page { get; set; }
        public int PageSize { get; set; }
    }
}
