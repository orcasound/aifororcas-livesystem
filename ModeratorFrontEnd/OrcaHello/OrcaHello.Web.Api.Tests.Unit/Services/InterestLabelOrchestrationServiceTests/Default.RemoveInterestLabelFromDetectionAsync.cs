namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class InterestLabelOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RemoveInterestLabelFromDetectionAsync_Expect()
        {
            Metadata metadata = new()
            {
                InterestLabel = "test"
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(metadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(true);

            _metadataServiceMock.Setup(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);


            InterestLabelRemovalResponse result = await _orchestrationService.RemoveInterestLabelFromDetectionAsync("id");

            Assert.IsNotNull(result);
            Assert.AreEqual("test", result.LabelRemoved);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()),
                Times.Once);
        }

        [TestMethod]
        public async Task Default_RemoveInterestLabelFromDetectionAsync_Expect_Rollback()
        {
            Metadata metadata = new()
            {
                InterestLabel = "test"
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(metadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(true);

            _metadataServiceMock.SetupSequence(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(false)
                .ReturnsAsync(true);

            try
            {
                InterestLabelRemovalResponse result = await _orchestrationService.RemoveInterestLabelFromDetectionAsync("id");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex is InterestLabelOrchestrationValidationException &&
                    ex.InnerException is DetectionNotInsertedException);
            }

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()),
                Times.Exactly(2));
        }
    }
}
