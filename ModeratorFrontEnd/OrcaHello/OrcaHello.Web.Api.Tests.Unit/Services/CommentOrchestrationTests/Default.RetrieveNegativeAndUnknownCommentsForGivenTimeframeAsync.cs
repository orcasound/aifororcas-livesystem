using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Comments;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class CommentOrchestrationTests
    {
        [TestMethod]
        public async Task Default_RetrieveNegativeAndUnknownCommentsForGivenTimeframeAsync_Expect()
        {
            var nullModeratedMetadata = CreateRandomMetadata();
            nullModeratedMetadata.DateModerated = null!;

            var expectedResults = new QueryableMetadataForTimeframe
            {
                QueryableRecords = (new List<Metadata> { CreateRandomMetadata(), nullModeratedMetadata }).AsQueryable(),
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                TotalCount = 1,
                Page = 1,
                PageSize = 10
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(expectedResults);

            CommentListResponse result = await _orchestrationService.RetrieveNegativeAndUnknownCommentsForGivenTimeframeAsync(DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

            Assert.AreEqual(expectedResults.QueryableRecords.Count(), result.Comments.Count);

            _metadataServiceMock.Verify(service =>
                 service.RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                   Times.Once);
        }
    }
}
