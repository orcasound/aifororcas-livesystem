namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredMetricsForModeratorAsync()
        {
            MetricsForModeratorResponse expectedResponse = new()
            {
                Negative = 1,
                Positive = 5,
                Unknown = 8,
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Moderator = "Moderator"
            };

            _moderatorServiceMock.Setup(service =>
                service.GetFilteredMetricsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                    .ReturnsAsync(expectedResponse);

            MetricsByDateRequest request = new()
            {
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
            };

            ModeratorMetricsItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredMetricsForModeratorAsync("Moderator", request);

            var positive = actualResponse.MetricsItemViews.Where(x => x.Name == "Positive").FirstOrDefault();

            Assert.AreEqual(expectedResponse.Positive, positive?.Value);

            _moderatorServiceMock.Verify(service =>
                service.GetFilteredMetricsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}