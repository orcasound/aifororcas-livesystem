namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataByIdAsync()
        {
            Metadata expectedResult = CreateRandomMetadata();

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataById(It.IsAny<string>()))
                    .ReturnsAsync(expectedResult);

            var result = await _metadataService.
                RetrieveMetadataByIdAsync(Guid.NewGuid().ToString());

            Assert.AreEqual(expectedResult.LocationName, result.LocationName);

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataById(It.IsAny<string>()),
                Times.Once);
        }
    }
}
