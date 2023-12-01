namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class TagViewServiceTests
    {
        private readonly Mock<ITagService> _tagServiceMock;
        private readonly Mock<IDetectionService> _detectionServiceMock;
        private readonly Mock<ILogger<TagViewService>> _loggerMock;

        private readonly ITagViewService _viewService;

        public TagViewServiceTests()
        {
            _tagServiceMock = new Mock<ITagService>();
            _detectionServiceMock = new Mock<IDetectionService>();
            _loggerMock = new Mock<ILogger<TagViewService>>();

            _viewService = new TagViewService(
                tagService: _tagServiceMock.Object,
                detectionService: _detectionServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _tagServiceMock.VerifyNoOtherCalls();
            _detectionServiceMock.VerifyNoOtherCalls();
        }
    }
}