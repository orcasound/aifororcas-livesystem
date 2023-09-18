namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class TagsControllerTests
    {
        private readonly Mock<ITagOrchestrationService> _orchestrationServiceMock;
        private readonly TagsController _controller;

        public TagsControllerTests()
        {
            _orchestrationServiceMock = new Mock<ITagOrchestrationService>();

            _controller = new TagsController(
                tagOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
