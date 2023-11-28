namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class ModeratorServiceTests
    {
        private readonly Mock<IDetectionAPIBroker> _apiBrokerMock;
        private readonly Mock<ILogger<ModeratorService>> _loggerMock;

        private readonly IModeratorService _service;

        public ModeratorServiceTests()
        {
            _apiBrokerMock = new Mock<IDetectionAPIBroker>();
            _loggerMock = new Mock<ILogger<ModeratorService>>();

            _service = new ModeratorService(
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
