namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class TagOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RemoveTagFromAllDetectionsAsync_Expect()
        {
            var tagToRemove = "TagToRemove";

            QueryableMetadata metadataWithTag = new()
            {
                QueryableRecords = (new List<Metadata>() { new(), new() }).AsQueryable(),
                TotalCount = 2
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataForTagAsync(It.IsAny<string>()))
                .ReturnsAsync(metadataWithTag);

            _metadataServiceMock.Setup(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);

            TagRemovalResponse result = await _orchestrationService.RemoveTagFromAllDetectionsAsync(tagToRemove);

            Assert.IsNotNull(result);
            Assert.AreEqual(tagToRemove, result.Tag);
            Assert.AreEqual(2, result.TotalMatching);
            Assert.AreEqual(2, result.TotalRemoved);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataForTagAsync(It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()),
                Times.Exactly(2));
        }
    }
}