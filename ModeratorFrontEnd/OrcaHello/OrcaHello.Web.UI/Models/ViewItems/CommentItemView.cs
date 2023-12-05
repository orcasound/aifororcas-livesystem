namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class CommentItemView
    {
        public string Id { get; set; } = null!;
        public string Comments { get; set; } = null!;
        public string LocationName { get; set; } = null!;
        public string Moderator { get; set; } = null!;
        public DateTime? Moderated { get; set; }
        public DateTime Timestamp { get; set; }
        public string AudioUri { get; set; } = null!;
        public string SpectrogramUri { get; set; } = null!;
        public bool IsCurrentlyPlaying { get; set; } = false;

        public static Func<Comment, CommentItemView> AsCommentItemView =>
            comment => new CommentItemView
            {
                Id = comment.Id,
                LocationName = comment.LocationName,
                Timestamp = comment.Timestamp,
                Comments = comment.Comments,
                Moderator = comment.Moderator,
                Moderated = comment.Moderated,
                AudioUri = comment.AudioUri,
                SpectrogramUri = comment.SpectrogramUri,
            };
    }
}
