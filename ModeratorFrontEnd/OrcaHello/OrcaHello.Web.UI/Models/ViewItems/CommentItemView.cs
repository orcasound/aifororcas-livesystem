namespace OrcaHello.Web.UI.Models
{
    public class CommentItemView
    {
        public string Id { get; set; }
        public string Comments { get; set; }
        public string LocationName { get; set; }
        public string Moderator { get; set; }
        public DateTime? Moderated { get; set; }
        public DateTime Timestamp { get; set; }
        public string AudioUri { get; set; }
        public string SpectrogramUri { get; set; }
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
