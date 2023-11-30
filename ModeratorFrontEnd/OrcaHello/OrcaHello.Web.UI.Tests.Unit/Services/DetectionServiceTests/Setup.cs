namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class DetectionServiceTests
    {
        private readonly Mock<IDetectionAPIBroker> _apiBrokerMock;
        private readonly Mock<ILogger<DetectionService>> _loggerMock;

        private readonly IDetectionService _service;

        public DetectionServiceTests()
        {
            _apiBrokerMock = new Mock<IDetectionAPIBroker>();
            _loggerMock = new Mock<ILogger<DetectionService>>();

            _service = new DetectionService(
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
