namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveNegativeAndUnknownMetadataForGivenTimeframeAndModeratorAsync()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new()
                {
                    CreateRandomMetadata()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveNegativeAndUnknownMetadataForGivenTimeframeAndModeratorAsync(fromDate, toDate, "Moderator", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
               Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveNegativeAndUnknownMetadataForGivenTimeframeAndModeratorAsync_ZeroPageAndPageSize()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new()
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveNegativeAndUnknownMetadataForGivenTimeframeAndModeratorAsync(fromDate, toDate, "Moderator", -1, -10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
               Times.Once);
        }
    }
}
