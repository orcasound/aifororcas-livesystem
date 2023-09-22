namespace OrcaHello.Web.UI.Services
{
    public interface IDetectionViewService
    {
        ValueTask<DetectionItemViewResponse> RetrieveFilteredAndPaginatedDetectionItemViewsAsync(DetectionFilterAndPagination options);
    }
}
