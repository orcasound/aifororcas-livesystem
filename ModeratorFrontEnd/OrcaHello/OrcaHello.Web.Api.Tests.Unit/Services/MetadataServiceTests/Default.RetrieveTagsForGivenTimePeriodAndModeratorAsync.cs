namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveTagsForGivenTimePeriodAndModeratorAsync()
        {
            var tags = new List<string>
            {
                "Tag1",
                "Tag2"
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetTagListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                    .ReturnsAsync(tags);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveTagsForGivenTimePeriodAndModeratorAsync(fromDate, toDate, "moderator");

            Assert.AreEqual(tags.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
            broker.GetTagListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
