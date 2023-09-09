namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class InterestLabelOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveAllInterestLabelsAsync_Expect()
        {
            var expectedResults = new QueryableInterestLabels
            {
                QueryableRecords = (new List<string> { "Label" }).AsQueryable(),
                TotalCount = 1
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveAllInterestLabelsAsync())
                .ReturnsAsync(expectedResults);

            InterestLabelListResponse result = await _orchestrationService.RetrieveAllInterestLabelsAsync();

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.InterestLabels.Count());

            _metadataServiceMock.Verify(service =>
                service.RetrieveAllInterestLabelsAsync(),
                Times.Once);
        }
    }
}
