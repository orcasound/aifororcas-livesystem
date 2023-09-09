namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForGivenTimeframeAndTagAsync()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new List<Metadata>
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetadataForGivenTimeframeAndTagAsync(fromDate, toDate, "tag", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForGivenTimeframeAndTagAsync_AndTags()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new List<Metadata>
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetadataForGivenTimeframeAndTagAsync(fromDate, toDate, "tag1,tag2", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForGivenTimeframeAndTagAsync_OrTags()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new List<Metadata>
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetadataForGivenTimeframeAndTagAsync(fromDate, toDate, "tag1|tag2", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForGivenTimeframeAndTagAsync_ZeroPageAndPageSize()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new List<Metadata>
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetadataForGivenTimeframeAndTagAsync(fromDate, toDate, "tag", -1, -10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataListByTimeframeAndTag(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<List<string>>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}