namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class ModeratorsControllerTests
    {
        private readonly Mock<IModeratorOrchestrationService> _orchestrationServiceMock;
        private readonly ModeratorsController _controller;

        public ModeratorsControllerTests()
        {
            _orchestrationServiceMock = new Mock<IModeratorOrchestrationService>();

            _controller = new ModeratorsController(
                moderatorOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
