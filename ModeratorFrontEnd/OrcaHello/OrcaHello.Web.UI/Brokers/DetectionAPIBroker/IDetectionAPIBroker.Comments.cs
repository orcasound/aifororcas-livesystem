namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<CommentListResponse> GetFilteredPositiveCommentsAsync(string queryString);
        ValueTask<CommentListResponse> GetFilteredNegativeAndUknownCommentsAsync(string queryString);
    }
}
