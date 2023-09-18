namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveDetectionByIdAsync_Expect()
        {
            var expectedMetadata = CreateRandomMetadata();

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedMetadata);

            Detection result = await _orchestrationService.RetrieveDetectionByIdAsync(Guid.NewGuid().ToString());

            Assert.AreEqual(expectedMetadata.LocationName, result.LocationName);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                    Times.Once);
        }
    }
}
