namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class HydrophoneViewServiceTests
    {
        private readonly Mock<IHydrophoneService> _hydrophoneServiceMock;
        private readonly Mock<ILogger<HydrophoneViewService>> _loggerMock;

        private readonly IHydrophoneViewService _viewService;

        public HydrophoneViewServiceTests()
        {
            _hydrophoneServiceMock = new Mock<IHydrophoneService>();
            _loggerMock = new Mock<ILogger<HydrophoneViewService>>();

            _viewService = new HydrophoneViewService(
                hydrophoneService: _hydrophoneServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _hydrophoneServiceMock.VerifyNoOtherCalls();
        }
    }
}
