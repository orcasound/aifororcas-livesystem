namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class DetectionViewServiceTests
    {
        private readonly Mock<IDetectionService> _detectionServiceMock;
        private readonly Mock<ITagService> _tagServiceMock;
        private readonly Mock<ILogger<DetectionViewService>> _loggerMock;

        private readonly IDetectionViewService _viewService;

        public DetectionViewServiceTests()
        {
            _detectionServiceMock = new Mock<IDetectionService>();
            _tagServiceMock = new Mock<ITagService>();
            _loggerMock = new Mock<ILogger<DetectionViewService>>();

            _viewService = new DetectionViewService(
                detectionService: _detectionServiceMock.Object,
                tagService: _tagServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _detectionServiceMock.VerifyNoOtherCalls();
            _tagServiceMock.VerifyNoOtherCalls();
        }
    }
}
