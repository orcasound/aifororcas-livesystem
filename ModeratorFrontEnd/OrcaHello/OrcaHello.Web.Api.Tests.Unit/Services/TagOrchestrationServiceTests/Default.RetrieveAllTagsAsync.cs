namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class TagOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveAllTagsAsync_Expect()
        {
            var expectedResults = new QueryableTags
            {
                QueryableRecords = (new List<string> { "Tag" }).AsQueryable(),
                TotalCount = 1
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveAllTagsAsync())
                .ReturnsAsync(expectedResults);

            TagListResponse result = await _orchestrationService.RetrieveAllTagsAsync();

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Tags.Count());

            _metadataServiceMock.Verify(service =>
                service.RetrieveAllTagsAsync(),
                Times.Once);
        }
    }
}
