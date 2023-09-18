namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RemoveMetadataByIdAndStateAsync()
        {
            _storageBrokerMock.Setup(broker =>
                broker.DeleteMetadataByIdAndState(It.IsAny<string>(), It.IsAny<string>()))
                    .ReturnsAsync(true);

            var result = await _metadataService.
                RemoveMetadataByIdAndStateAsync(Guid.NewGuid().ToString(), "Unreviewed");

            Assert.IsTrue(result);

            _storageBrokerMock.Verify(broker =>
                broker.DeleteMetadataByIdAndState(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
