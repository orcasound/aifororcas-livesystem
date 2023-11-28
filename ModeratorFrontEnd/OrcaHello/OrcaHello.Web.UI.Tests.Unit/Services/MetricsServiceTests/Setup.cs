namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class MetricsServiceTest
    {
        private readonly Mock<IDetectionAPIBroker> _apiBrokerMock;
        private readonly Mock<ILogger<MetricsService>> _loggerMock;

        private readonly IMetricsService _service;

        public MetricsServiceTest()
        {
            _apiBrokerMock = new Mock<IDetectionAPIBroker>();
            _loggerMock = new Mock<ILogger<MetricsService>>();

            _service = new MetricsService(
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
