using Moq;
using OrcaHello.Web.Api.Models.Configurations;
using OrcaHello.Web.Shared.Models.Hydrophones;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class HydrophoneOrchestrationServiceTests
    {
        [TestMethod]
        public void Default_RetrieveHydrophoneLocations_Expect()
        {
            HydrophoneLocation location = new HydrophoneLocation
            {
                Name = "Test",
                Id = "source_guid"
            };

            _appSettingsMock.Setup(g =>
                g.HydrophoneLocations).
                Returns(new List<HydrophoneLocation> { location });

            HydrophoneListResponse result = _orchestrationService.
                RetrieveHydrophoneLocations();

            Assert.AreEqual(1, result.Count);

            _appSettingsMock.Verify(service =>
                service.HydrophoneLocations,
                Times.Exactly(2));
        }
    }
}
