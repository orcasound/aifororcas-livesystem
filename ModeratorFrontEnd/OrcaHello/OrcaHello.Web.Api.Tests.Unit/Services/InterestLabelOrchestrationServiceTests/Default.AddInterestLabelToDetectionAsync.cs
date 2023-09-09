namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class InterestLabelOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_AddInterestLabelToDetectionAsync_Expect()
        {
            Metadata metadata = new();

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(metadata);

            _metadataServiceMock.Setup(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);

            InterestLabelAddResponse result = await _orchestrationService.AddInterestLabelToDetectionAsync("id", "label");

            Assert.IsNotNull(result);
            Assert.AreEqual("label", result.LabelAdded);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()),
                Times.Once);
        }

    }
}
