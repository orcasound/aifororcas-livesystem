namespace OrcaHello.Web.Api.Services
{
    public interface ITagOrchestrationService
    {
        ValueTask<TagListResponse> RetrieveAllTagsAsync();
        ValueTask<TagListForTimeframeResponse> RetrieveTagsForGivenTimePeriodAsync(DateTime? fromDate, DateTime? toDate);
        ValueTask<TagRemovalResponse> RemoveTagFromAllDetectionsAsync(string tagToRemove);
        ValueTask<TagReplaceResponse> ReplaceTagInAllDetectionsAsync(ReplaceTagRequest request);
    }
}
