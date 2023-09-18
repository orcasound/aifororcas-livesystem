using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Metrics;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetricsOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveMetricsForGivenTimeframeAsync_Expect()
        {
            MetricsSummaryForTimeframe expectedResult = new MetricsSummaryForTimeframe
            {
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                QueryableRecords = (new List<MetricResult>
                {
                    new MetricResult { State = "Positive", Count = 1}
                }).AsQueryable()
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                .ReturnsAsync(expectedResult);

            MetricsResponse result = await _orchestrationService.RetrieveMetricsForGivenTimeframeAsync(DateTime.Now, DateTime.Now.AddDays(1));

            Assert.AreEqual(expectedResult.QueryableRecords.Where(x => x.State == "Positive").Select(x => x.Count).FirstOrDefault(), result.Positive);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                    Times.Once);
        }
    }
}
