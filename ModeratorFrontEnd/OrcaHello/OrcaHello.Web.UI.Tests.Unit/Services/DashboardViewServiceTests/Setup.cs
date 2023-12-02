namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class DashboardViewServiceTests
    {
        private readonly Mock<IDetectionService> _detectionServiceMock;
        private readonly Mock<ITagService> _tagServiceMock;
        private readonly Mock<IMetricsService> _metricsServiceMock;
        private readonly Mock<ICommentService> _commentServiceMock;
        private readonly Mock<IModeratorService> _moderatorServiceMock;

        private readonly Mock<ILogger<DashboardViewService>> _loggerMock;

        private readonly IDashboardViewService _viewService;

        public DashboardViewServiceTests()
        {
            _detectionServiceMock = new Mock<IDetectionService>();
            _tagServiceMock = new Mock<ITagService>();
            _metricsServiceMock = new Mock<IMetricsService>();
            _commentServiceMock = new Mock<ICommentService>();
            _moderatorServiceMock = new Mock<IModeratorService>();

            _loggerMock = new Mock<ILogger<DashboardViewService>>();

            _viewService = new DashboardViewService(
                detectionService: _detectionServiceMock.Object,
                tagService: _tagServiceMock.Object,
                metricsService: _metricsServiceMock.Object,
                commentService: _commentServiceMock.Object,
                moderatorService: _moderatorServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _detectionServiceMock.VerifyNoOtherCalls();
            _tagServiceMock.VerifyNoOtherCalls();
            _metricsServiceMock.VerifyNoOtherCalls();
            _commentServiceMock.VerifyNoOtherCalls();
            _moderatorServiceMock.VerifyNoOtherCalls();
        }
    }
}