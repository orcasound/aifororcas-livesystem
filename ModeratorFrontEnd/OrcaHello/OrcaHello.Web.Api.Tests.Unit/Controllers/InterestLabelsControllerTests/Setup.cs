namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class InterestLabelsControllerTests
    {
        private readonly Mock<IInterestLabelOrchestrationService> _orchestrationServiceMock;
        private readonly InterestLabelsController _controller;

        public InterestLabelsControllerTests()
        {
            _orchestrationServiceMock = new Mock<IInterestLabelOrchestrationService>();

            _controller = new InterestLabelsController(
                interestLabelOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
