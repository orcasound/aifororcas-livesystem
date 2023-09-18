namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveUnreviewedMetadataForGivenTimeframeAsync()
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
                broker.GetUnreviewedMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveUnreviewedMetadataForGivenTimeframeAsync(fromDate, toDate, 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                  broker.GetUnreviewedMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveUnreviewedMetadataForGivenTimeframeAsync_ZeroPageAndPageSize()
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
                broker.GetUnreviewedMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveUnreviewedMetadataForGivenTimeframeAsync(fromDate, toDate, -1, -10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                  broker.GetUnreviewedMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }
    }
}
