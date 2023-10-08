namespace OrcaHello.Web.UI.Services
{
    public interface IDetectionViewService
    {
        ValueTask<DetectionItemViewResponse> RetrieveFilteredAndPaginatedDetectionItemViewsAsync(DetectionFilterAndPagination options);
        ValueTask<ModerateDetectionsResponse> ModerateDetectionsAsync(List<string> ids, string state, string moderator, string comments, string tags);
    }
}
