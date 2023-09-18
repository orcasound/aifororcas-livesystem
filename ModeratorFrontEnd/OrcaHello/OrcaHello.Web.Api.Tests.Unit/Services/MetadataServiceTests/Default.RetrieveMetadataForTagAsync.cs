namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForTagAsync()
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
                broker.GetAllMetadataListByTag(It.IsAny<string>()))
                    .ReturnsAsync(expectedResult);

            var result = await _metadataService.
                RetrieveMetadataForTagAsync("tag");

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetAllMetadataListByTag(It.IsAny<string>()),
                Times.Once);
        }
    }
}
