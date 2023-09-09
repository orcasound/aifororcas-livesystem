namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class InterestLabelOrchestrationServiceTests
    {
        private readonly Mock<IMetadataService> _metadataServiceMock;
        private readonly Mock<ILogger<InterestLabelOrchestrationService>> _loggerMock;

        private readonly IInterestLabelOrchestrationService _orchestrationService;

        public InterestLabelOrchestrationServiceTests()
        {
            _metadataServiceMock = new Mock<IMetadataService>();
            _loggerMock = new Mock<ILogger<InterestLabelOrchestrationService>>();

            _orchestrationService = new InterestLabelOrchestrationService(
                metadataService: _metadataServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _metadataServiceMock.VerifyNoOtherCalls();
        }
    }
}