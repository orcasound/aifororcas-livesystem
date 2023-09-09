using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Hydrophones;
using static OrcaHello.Web.Api.Services.HydrophoneOrchestrationService;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class HydrophoneOrchestrationServiceTests
    {
        [TestMethod]
        public void TryCatch_DetectionListResponse_Expect_Exception()
        {
            var wrapper = new HydrophoneOrchestrationServiceWrapper();
            var delegateMock = new Mock<ReturningGenericFunction<HydrophoneListResponse>>();

            delegateMock
               .SetupSequence(p => p())

           .Throws(new InvalidHydrophoneOrchestrationException())

           .Throws(new Exception());

            Assert.ThrowsException<HydrophoneOrchestrationValidationException>(() =>
                wrapper.TryCatch(delegateMock.Object));

            Assert.ThrowsException<HydrophoneOrchestrationServiceException>(() =>
                wrapper.TryCatch(delegateMock.Object));
        }
    }
}
