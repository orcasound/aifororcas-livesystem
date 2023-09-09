namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class HydrophonesControllerTests
    {
        private readonly Mock<IHydrophoneOrchestrationService> _orchestrationServiceMock;
        private readonly HydrophonesController _controller;

        public HydrophonesControllerTests()
        {
            _orchestrationServiceMock = new Mock<IHydrophoneOrchestrationService>();

            _controller = new HydrophonesController(
                hydrophoneOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
