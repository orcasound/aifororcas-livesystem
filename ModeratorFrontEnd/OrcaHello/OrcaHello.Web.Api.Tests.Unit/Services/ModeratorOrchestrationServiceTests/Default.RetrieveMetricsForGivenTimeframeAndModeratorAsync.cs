using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Moderators;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class ModeratorOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveMetricsForGivenTimeframeAndModeratorAsync_Expect()
        {
            MetricsSummaryForTimeframeAndModerator expectedResult = new MetricsSummaryForTimeframeAndModerator
            {
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                QueryableRecords = (new List<MetricResult>
                {
                    new MetricResult { State = "Positive", Count = 1}
                }).AsQueryable(),
                Moderator = "Moderator"
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResult);

            MetricsForModeratorResponse result = await _orchestrationService.RetrieveMetricsForGivenTimeframeAndModeratorAsync(DateTime.Now, DateTime.Now.AddDays(1), "Moderator");

            Assert.AreEqual(expectedResult.QueryableRecords.Where(x => x.State == "Positive").Select(x => x.Count).FirstOrDefault(), result.Positive);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                    Times.Once);
        }
    }
}
