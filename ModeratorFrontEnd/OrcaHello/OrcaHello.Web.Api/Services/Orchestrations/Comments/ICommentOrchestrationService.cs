namespace OrcaHello.Web.Api.Services
{
    public interface ICommentOrchestrationService
    {
        ValueTask<CommentListResponse> RetrievePositiveCommentsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate, int page, int pageSize);
        ValueTask<CommentListResponse> RetrieveNegativeAndUnknownCommentsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate, int page, int pageSize);
    }
}
