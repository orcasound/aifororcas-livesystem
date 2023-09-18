namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class HydrophoneOrchestrationServiceTests
    {
        private readonly Mock<IHydrophoneService> _hydrophoneServiceMock;
        private readonly Mock<ILogger<HydrophoneOrchestrationService>> _loggerMock;

        private readonly IHydrophoneOrchestrationService _orchestrationService;

        public HydrophoneOrchestrationServiceTests()
        {
            _loggerMock = new Mock<ILogger<HydrophoneOrchestrationService>>();
            _hydrophoneServiceMock = new Mock<IHydrophoneService>();

            _orchestrationService = new HydrophoneOrchestrationService(
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