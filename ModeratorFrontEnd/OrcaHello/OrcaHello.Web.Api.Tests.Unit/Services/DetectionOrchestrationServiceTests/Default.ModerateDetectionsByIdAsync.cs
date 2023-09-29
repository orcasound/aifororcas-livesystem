namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {

        [TestMethod]
        public async Task Default_ModerateDetectionsById_Expect()
        {
            var id = Guid.NewGuid().ToString();

            var storedMetadata = CreateRandomMetadata();
            storedMetadata.Id = id;

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(storedMetadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(true);

            _metadataServiceMock.Setup(service =>
                service.AddMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);

            ModerateDetectionsRequest request = new()
            {
                Ids = new List<string> { id },
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            ModerateDetectionsResponse result = await _orchestrationService.ModerateDetectionsByIdAsync(request);

            Assert.AreEqual(storedMetadata.Id, request.Ids.FirstOrDefault());

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
        public async Task ModerateDetectionsById_ExpectNotFound()
        {
            Metadata blankMetadata = null!;

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(blankMetadata);

            ModerateDetectionsRequest request = new()
            {
                Ids = new List<string> { Guid.NewGuid().ToString() },
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            ModerateDetectionsResponse result = await _orchestrationService.ModerateDetectionsByIdAsync(request);

            Assert.AreEqual(1, result.IdsNotFound.Count);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                    Times.Once);
        }

        [TestMethod]
        public async Task ModerateDetectionsById_ExpectNotDeleted()
        {
            var id = Guid.NewGuid().ToString();

            var storedMetadata = CreateRandomMetadata();
            storedMetadata.Id = id;

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()))
                .ReturnsAsync(storedMetadata);

            _metadataServiceMock.Setup(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(false);

            ModerateDetectionsRequest request = new()
            {
                Ids = new List<string> { Guid.NewGuid().ToString() },
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            ModerateDetectionsResponse result = await _orchestrationService.ModerateDetectionsByIdAsync(request);

            Assert.AreEqual(1, result.IdsUnsuccessfullyUpdated.Count);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataByIdAsync(It.IsAny<string>()),
                    Times.Once);

            _metadataServiceMock.Verify(service =>
                service.RemoveMetadataByIdAndStateAsync(It.IsAny<string>(), It.IsAny<string>()),
                    Times.Once);
        }

        [TestMethod]
        public async Task ModerateDetectionsById_ExpectNoUpdate()
        {
            var id = Guid.NewGuid().ToString();

            var storedMetadata = CreateRandomMetadata();
            storedMetadata.Id = id;

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

            ModerateDetectionsRequest request = new()
            {
                Ids = new List<string> { id },
                State = DetectionState.Positive.ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1", "Tag2" }
            };

            ModerateDetectionsResponse result = await _orchestrationService.ModerateDetectionsByIdAsync(request);

            Assert.AreEqual(1, result.IdsUnsuccessfullyUpdated.Count);

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
