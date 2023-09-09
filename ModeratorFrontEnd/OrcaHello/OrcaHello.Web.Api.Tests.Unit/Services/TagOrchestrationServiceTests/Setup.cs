namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class TagOrchestrationServiceTests
    {
        private readonly Mock<IMetadataService> _metadataServiceMock;
        private readonly Mock<ILogger<TagOrchestrationService>> _loggerMock;

        private readonly ITagOrchestrationService _orchestrationService;

        public TagOrchestrationServiceTests()
        {
            _metadataServiceMock = new Mock<IMetadataService>();
            _loggerMock = new Mock<ILogger<TagOrchestrationService>>();

            _orchestrationService = new TagOrchestrationService(
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
