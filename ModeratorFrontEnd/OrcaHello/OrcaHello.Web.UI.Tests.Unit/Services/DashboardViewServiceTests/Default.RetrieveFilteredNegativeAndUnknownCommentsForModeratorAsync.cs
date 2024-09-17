namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredNegativeAndUnknownCommentsForModeratorAsync()
        {
            CommentListForModeratorResponse expectedResponse = new()
            {
                Comments = new()
                {
                    new() { Comments = "Comments 1" },
                    new() { Comments = "Comments 2" }
                },
                TotalCount = 20,
                Count = 2,
                Moderator = "Moderator"
            };

            _moderatorServiceMock.Setup(service =>
                service.GetFilteredNegativeAndUknownCommentsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResponse);

            PaginatedCommentsByDateRequest request = new()
            {
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Page = 1,
                PageSize = 10
            };

            ModeratorCommentItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredNegativeAndUnknownCommentsForModeratorAsync("Moderator", request);

            Assert.AreEqual(expectedResponse.TotalCount, actualResponse.Count);

            _moderatorServiceMock.Verify(service =>
                service.GetFilteredNegativeAndUknownCommentsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}
