namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForInterestLabelAsync()
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
                broker.GetAllMetadataListByInterestLabel(It.IsAny<string>()))
                    .ReturnsAsync(expectedResult);

            var result = await _metadataService.
                RetrieveMetadataForInterestLabelAsync("test");

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetAllMetadataListByInterestLabel(It.IsAny<string>()),
                Times.Once);
        }
    }
}
