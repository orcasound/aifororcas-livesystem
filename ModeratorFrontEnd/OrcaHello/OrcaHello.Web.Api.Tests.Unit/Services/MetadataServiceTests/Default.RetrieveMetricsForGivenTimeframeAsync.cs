namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetricsForGivenTimeframeAsync()
        {
            List<MetricResult> expectedResults = new()
            {
                new() { State = "Unreviewed", Count = 1 },
                new() { State = "Positive", Count = 5 },
                new() { State = "Negative", Count = 10 },
                new() { State = "Unknown", Count = 2 },
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetricsListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                    .ReturnsAsync(expectedResults);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetricsForGivenTimeframeAsync(fromDate, toDate);

            Assert.AreEqual(4, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetricsListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}
