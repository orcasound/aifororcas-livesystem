namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class TagOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_ReplaceTagInAllDetectionsAsync_Expect()
        {
            var oldTag = "OldTag";
            var newTag = "NewTag";

            QueryableMetadata metadataWithTag = new()
            {
                QueryableRecords = (new List<Metadata>() { new() { Tags = new List<string>() { "OldTag" } }, new() { Tags = new List<string>() { "OldTag" } } }).AsQueryable(),
                TotalCount = 2
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataForTagAsync(It.IsAny<string>()))
                .ReturnsAsync(metadataWithTag);

            _metadataServiceMock.Setup(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()))
                .ReturnsAsync(true);

            TagReplaceResponse result = await _orchestrationService.ReplaceTagInAllDetectionsAsync(new ReplaceTagRequest { OldTag = oldTag, NewTag = newTag });

            Assert.IsNotNull(result);
            Assert.AreEqual(oldTag, result.OldTag);
            Assert.AreEqual(newTag, result.NewTag);
            Assert.AreEqual(2, result.TotalMatching);
            Assert.AreEqual(2, result.TotalReplaced);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataForTagAsync(It.IsAny<string>()),
                Times.Once);

            _metadataServiceMock.Verify(service =>
                service.UpdateMetadataAsync(It.IsAny<Metadata>()),
                Times.Exactly(2));
        }
    }
}