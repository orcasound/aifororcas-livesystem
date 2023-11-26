namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class CommentServiceTests
    {
        private readonly Mock<IDetectionAPIBroker> _apiBrokerMock;
        private readonly Mock<ILogger<CommentService>> _loggerMock;

        private readonly ICommentService _service;

        public CommentServiceTests()
        {
            _apiBrokerMock = new Mock<IDetectionAPIBroker>();
            _loggerMock = new Mock<ILogger<CommentService>>();

            _service = new CommentService(
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
