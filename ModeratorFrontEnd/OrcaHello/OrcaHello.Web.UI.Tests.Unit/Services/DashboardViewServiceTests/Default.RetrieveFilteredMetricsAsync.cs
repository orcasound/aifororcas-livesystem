namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredMetricsAsync()
        {
            MetricsResponse expectedResponse = new()
            {
                Negative = 1,
                Positive = 5,
                Unknown = 8,
                Unreviewed = 9,
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
            };

            _metricsServiceMock.Setup(service =>
                service.RetrieveFilteredMetricsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                    .ReturnsAsync(expectedResponse);

            MetricsByDateRequest request = new()
            {
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
            };

            MetricsItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredMetricsAsync(request);

            var unreviewed = actualResponse.MetricsItemViews.Where(x => x.Name == "Unreviewed").FirstOrDefault();

            Assert.AreEqual(expectedResponse.Unreviewed, unreviewed?.Value);

            _metricsServiceMock.Verify(service =>
                service.RetrieveFilteredMetricsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}