namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_UpdateMetadataAsync()
        {
            _storageBrokerMock.Setup(broker =>
                broker.UpdateMetadataInPartition(It.IsAny<Metadata>()))
                    .ReturnsAsync(true);

            Metadata metadata = CreateRandomMetadata();

            var result = await _metadataService.
                UpdateMetadataAsync(metadata);

            Assert.IsTrue(result);

            _storageBrokerMock.Verify(broker =>
                broker.UpdateMetadataInPartition(It.IsAny<Metadata>()),
                Times.Once);
        }
    }
}
