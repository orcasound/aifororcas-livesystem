namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {

        [TestMethod]
        public async Task Default_ModerateDetectionById_Expect()
        {
            var storedMetadata = CreateRandomMetadata();

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(storedMetadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(true);

            _metadataServiceMock.Setup(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);

            ModerateDetectionRequest request = new()
            {
                Id = Guid.NewGuid().ToString(),
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            Detection result = await _orchestrationService.ModerateDetectionByIdAsync(request.Id, request);

            Assert.AreEqual(storedMetadata.LocationName, result.LocationName);

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
        public async Task Error_ModerateDetectionById_ExpectException()
        {
            var storedMetadata = CreateRandomMetadata();

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(storedMetadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(true);

            _metadataServiceMock.SetupSequence(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(false)
                .ReturnsAsync(true);

            ModerateDetectionRequest request = new()
            {
                Id = Guid.NewGuid().ToString(),
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            try
            {

                Detection result = await _orchestrationService.ModerateDetectionByIdAsync(request.Id, request);
            } 
            catch(Exception ex)
            {
                Assert.IsTrue(ex is DetectionOrchestrationValidationException &&
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
