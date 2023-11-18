using static System.Runtime.InteropServices.JavaScript.JSType;

namespace OrcaHello.Web.UI.Services
{
    // HydrophoneViewService implements the presentation layer of the software application,
    // according to the Standard. It is responsible for retrieving and displaying hydrophone data to the user,
    // using the HydrophoneService as a dependency and the LoggingUtilities as a helper.
    public partial class HydrophoneViewService : IHydrophoneViewService
    {
        // _hydrophoneService is used to interact with the hydrophone broker and perform business logic operations.
        private readonly IHydrophoneService _hydrophoneService;

        // _logger to log information, errors, and exceptions that occur during the execution of the methods.
        private readonly ILogger<HydrophoneViewService> _logger;

        public HydrophoneViewService(IHydrophoneService hydrophoneService,
            ILogger<HydrophoneViewService> logger)
        {
            _hydrophoneService = hydrophoneService;
            _logger = logger;
        }

        // Returns a list of hydrophone views ordered by name after retrieving and mapping them from the HydrophoneService.
        public ValueTask<List<HydrophoneItemView>> RetrieveAllHydrophoneViewsAsync() =>
        TryCatch(async () =>
        {
            List<Hydrophone> hydrophones =
                await _hydrophoneService.RetrieveAllHydrophonesAsync();

            return hydrophones.Select(AsHydrophoneView).OrderBy(x => x.Name).ToList();
        });

        // Maps a hydrophone to a hydrophone view for presentation.
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