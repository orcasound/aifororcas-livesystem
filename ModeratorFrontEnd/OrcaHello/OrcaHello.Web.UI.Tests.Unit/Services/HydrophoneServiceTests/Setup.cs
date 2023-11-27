namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class HydrophoneServiceTests
    {
        private readonly Mock<IDetectionAPIBroker> _apiBrokerMock;
        private readonly Mock<ILogger<HydrophoneService>> _loggerMock;

        private readonly IHydrophoneService _service;

        public HydrophoneServiceTests()
        {
            _apiBrokerMock = new Mock<IDetectionAPIBroker>();
            _loggerMock = new Mock<ILogger<HydrophoneService>>();

            _service = new HydrophoneService(
                apiBroker: _apiBrokerMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _apiBrokerMock.VerifyNoOtherCalls();
        }
    }
}
