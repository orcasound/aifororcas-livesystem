namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrievePositiveMetadataForGivenTimeframeAndModeratorAsync()
        {
            ListMetadataAndCount expectedResult = new ListMetadataAndCount
            {
                PaginatedRecords = new List<Metadata>
                {
                    CreateRandomMetadata()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetPositiveMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrievePositiveMetadataForGivenTimeframeAndModeratorAsync(fromDate, toDate, "Moderator", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetPositiveMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrievePositiveMetadataForGivenTimeframeAndModeratorAsync_ZeroPageAndPageSize()
        {
            ListMetadataAndCount expectedResult = new ListMetadataAndCount
            {
                PaginatedRecords = new List<Metadata>
                {
                    new Metadata()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetPositiveMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrievePositiveMetadataForGivenTimeframeAndModeratorAsync(fromDate, toDate, "Moderator", -1, -10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetPositiveMetadataListByTimeframeAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }
    }
}
