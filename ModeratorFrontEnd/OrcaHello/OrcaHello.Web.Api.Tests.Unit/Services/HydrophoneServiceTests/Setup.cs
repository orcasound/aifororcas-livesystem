namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class HydrophoneServiceTests
    {
        private readonly Mock<IHydrophoneBroker> _hydrophoneBrokerMock;
        private readonly Mock<ILogger<HydrophoneService>> _loggerMock;

        private readonly IHydrophoneService _hydrophoneService;

        public HydrophoneServiceTests()
        {
            _hydrophoneBrokerMock = new Mock<IHydrophoneBroker>();
            _loggerMock = new Mock<ILogger<HydrophoneService>>();

            _hydrophoneService = new HydrophoneService(
                hydrophoneBroker: _hydrophoneBrokerMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _hydrophoneBrokerMock.VerifyNoOtherCalls();
        }
    }
}
