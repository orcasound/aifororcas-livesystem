namespace OrcaHello.Web.UI.Services
{
    public interface IDetectionViewService
    {
        ValueTask<DetectionItemViewResponse> RetrieveFilteredAndPaginatedDetectionItemViewsAsync(PaginatedDetectionsByStateRequest options);
        ValueTask<ModerateDetectionsResponse> ModerateDetectionsAsync(List<string> ids, string state, string moderator, string comments, string tags);
        ValueTask<List<string>> RetrieveAllTagsAsync();
        ValueTask<DetectionItemView> RetrieveDetectionAsync(string id);
    }
}
