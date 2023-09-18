namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_AddMetadataAsync()
        {
            _storageBrokerMock.Setup(broker =>
                broker.InsertMetadata(It.IsAny<Metadata>()))
                    .ReturnsAsync(true);

            Metadata newRecord = CreateRandomMetadata();

            var result = await _metadataService.
                AddMetadataAsync(newRecord);

            Assert.IsTrue(result);

            _storageBrokerMock.Verify(broker =>
                broker.InsertMetadata(It.IsAny<Metadata>()),
                Times.Once);
        }
    }
}
