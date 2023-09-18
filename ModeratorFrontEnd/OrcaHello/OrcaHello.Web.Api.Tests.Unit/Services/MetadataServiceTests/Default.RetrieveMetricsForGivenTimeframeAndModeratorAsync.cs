namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetricsForGivenTimeframeAndModeratorAsync()
        {
            List<MetricResult> expectedResults = new()
            {
                new() { State = "Positive", Count = 5 },
                new() { State = "Negative", Count = 10 },
                new() { State = "Unknown", Count = 2 },
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetricsListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                    .ReturnsAsync(expectedResults);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetricsForGivenTimeframeAndModeratorAsync(fromDate, toDate, "Moderator");

            Assert.AreEqual(3, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetricsListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
