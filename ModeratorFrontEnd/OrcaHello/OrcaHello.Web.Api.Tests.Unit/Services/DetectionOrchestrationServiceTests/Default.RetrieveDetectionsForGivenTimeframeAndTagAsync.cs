namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveDetectionsForGivenTimeframeAndTagAsync_Expect()
        {
            var nullModeratedMetadata = CreateRandomMetadata();
            nullModeratedMetadata.DateModerated = null!;

            var expectedResults = new QueryableMetadataForTimeframeAndTag
            {
                QueryableRecords = (new List<Metadata> { CreateRandomMetadata(), nullModeratedMetadata }).AsQueryable(),
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                TotalCount = 1,
                Tag = "tag",
                Page = 1,
                PageSize = 10
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(expectedResults);

            DetectionListForTagResponse result = await _orchestrationService.RetrieveDetectionsForGivenTimeframeAndTagAsync(DateTime.Now, DateTime.Now.AddDays(1), "tag", 1, 10);

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Detections.Count);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }
    }
}
