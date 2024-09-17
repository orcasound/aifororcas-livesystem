namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class HydrophoneViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllHydrophoneViewsAsync()
        {
            List<Hydrophone> expectedResponse = new()
            {
                new() { Name = "Hydrophone #1" }
            };

            _hydrophoneServiceMock.Setup(broker =>
                broker.RetrieveAllHydrophonesAsync())
                .ReturnsAsync(expectedResponse);

            List<HydrophoneItemView> actualResponse =
                await _viewService.RetrieveAllHydrophoneViewsAsync();

            Assert.AreEqual(expectedResponse.Count(), actualResponse.Count());

            _hydrophoneServiceMock.Verify(broker =>
                broker.RetrieveAllHydrophonesAsync(),
                Times.Once);
        }
    }
}
