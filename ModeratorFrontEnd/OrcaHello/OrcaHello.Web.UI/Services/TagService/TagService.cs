namespace OrcaHello.Web.UI.Services
{
    public partial class TagService : ITagService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<TagService> _logger;

        public TagService(
            IDetectionAPIBroker apiBroker,
            ILogger<TagService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        public ValueTask<List<string>> RetrieveAllTagsAsync() =>
        TryCatch(async () =>
        {
            TagListResponse response = await _apiBroker.GetAllTagsAsync();
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response.Tags;
        });

        public ValueTask<TagListForTimeframeResponse> RetrieveFilteredTagsAsync(DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () =>
        {
            // ValidateRequiredProperties();

            var queryString = $"fromDate={fromDate.Value.ToString()}&toDate={toDate.Value.ToString()}";

            TagListForTimeframeResponse response = await _apiBroker.GetFilteredTagsAsync(queryString);
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response;
        });
    }
}
