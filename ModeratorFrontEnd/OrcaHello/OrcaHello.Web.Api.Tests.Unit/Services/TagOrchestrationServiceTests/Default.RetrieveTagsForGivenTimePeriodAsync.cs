namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class TagOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveTagsForGivenTimeframeAsync_Expect()
        {
            var expectedResults = new QueryableTagsForTimeframe
            {
                QueryableRecords = (new List<string> { "Tag" }).AsQueryable(),
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                TotalCount = 1
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveTagsForGivenTimePeriodAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                .ReturnsAsync(expectedResults);

            TagListResponse result = await _orchestrationService.RetrieveTagsForGivenTimePeriodAsync(DateTime.Now, DateTime.Now.AddDays(1));

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Tags.Count());

            _metadataServiceMock.Verify(service =>
                service.RetrieveTagsForGivenTimePeriodAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}
