namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class ModeratorServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_GetFilteredMetricsForModeratorAsync()
        {
            MetricsForModeratorResponse expectedResponse = new()
            {
                Positive = 2,
                Negative = 3,
                Unknown = 1
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredMetricsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            MetricsForModeratorResponse actualResponse = await _service.GetFilteredMetricsForModeratorAsync("John Smith", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow);

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredMetricsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}