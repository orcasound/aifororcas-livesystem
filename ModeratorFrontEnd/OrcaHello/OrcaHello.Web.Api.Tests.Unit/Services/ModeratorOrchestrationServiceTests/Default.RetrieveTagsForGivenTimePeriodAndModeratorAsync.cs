using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Tags;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class ModeratorOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveTagsForGivenTimeframeAndModeratorAsync_Expect()
        {
            var expectedResults = new QueryableTagsForTimeframeAndModerator
            {
                QueryableRecords = (new List<string> { "Tag" }).AsQueryable(),
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                TotalCount = 1,
                Moderator = "Moderator"
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveTagsForGivenTimePeriodAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResults);

            TagListResponse result = await _orchestrationService.RetrieveTagsForGivenTimePeriodAndModeratorAsync(DateTime.Now, DateTime.Now.AddDays(1), "Moderator");

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Tags.Count());

            _metadataServiceMock.Verify(service =>
                service.RetrieveTagsForGivenTimePeriodAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
