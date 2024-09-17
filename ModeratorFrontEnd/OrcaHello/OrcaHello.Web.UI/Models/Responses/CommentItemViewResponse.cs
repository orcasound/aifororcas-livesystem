namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class CommentItemViewResponse
    {
        public int Count { get; set; }
        public List<CommentItemView> CommentItemViews { get; set; } = null!;
    }
}
