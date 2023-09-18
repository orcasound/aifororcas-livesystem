namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class CommentsControllerTests
    {
        private readonly Mock<ICommentOrchestrationService> _orchestrationServiceMock;
        private readonly CommentsController _controller;

        public CommentsControllerTests()
        {
            _orchestrationServiceMock = new Mock<ICommentOrchestrationService>();

            _controller = new CommentsController(
                commentOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
