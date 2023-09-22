namespace OrcaHello.Web.UI.Services
{
    public partial class HydrophoneViewService : IHydrophoneViewService
    {
        private readonly IHydrophoneService _hydrophoneService;
        private readonly ILogger<HydrophoneViewService> _logger;

        public HydrophoneViewService(IHydrophoneService hydrophoneService,
            ILogger<HydrophoneViewService> logger)
        {
            _hydrophoneService = hydrophoneService;
            _logger = logger;
        }

        public ValueTask<List<HydrophoneItemView>> RetrieveAllHydrophoneViewsAsync() =>
        TryCatch(async () =>
        {
            List<Hydrophone> hydrophones =
                await _hydrophoneService.RetrieveAllHydrophonesAsync();

            return hydrophones.Select(AsHydrophoneView).OrderBy(x => x.Name).ToList();
        });

        private static Func<Hydrophone, HydrophoneItemView> AsHydrophoneView =>
            hydrophone => new HydrophoneItemView
            {
                Id = hydrophone.Id,
                Name = hydrophone.Name,
                Longitude = hydrophone.Longitude,
                Latitude = hydrophone.Latitude,
                ImageUrl = hydrophone.ImageUrl,
                IntroHtml = hydrophone.IntroHtml
            };
    }
}