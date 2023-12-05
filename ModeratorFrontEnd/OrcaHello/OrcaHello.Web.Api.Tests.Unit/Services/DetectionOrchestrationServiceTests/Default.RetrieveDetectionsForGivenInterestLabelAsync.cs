namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveDetectionsForGivenInterestLabelAsync_Expect()
        {
            var nullModeratedMetadata = CreateRandomMetadata();
            nullModeratedMetadata.DateModerated = null!;

            var expectedResults = new QueryableMetadata
            {
                QueryableRecords = (new List<Metadata> { CreateRandomMetadata(), nullModeratedMetadata }).AsQueryable(),
                TotalCount = 2,
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveMetadataForInterestLabelAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResults);

            DetectionListForInterestLabelResponse result = await _orchestrationService.RetrieveDetectionsForGivenInterestLabelAsync("label");

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Detections.Count);

            _metadataServiceMock.Verify(service =>
                service.RetrieveMetadataForInterestLabelAsync(It.IsAny<string>()),
                    Times.Once);
        }
    }
}
