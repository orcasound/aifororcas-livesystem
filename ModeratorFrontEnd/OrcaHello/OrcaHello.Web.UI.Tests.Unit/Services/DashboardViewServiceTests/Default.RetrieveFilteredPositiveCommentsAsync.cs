namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredPositiveCommentsAsync()
        {
            CommentListResponse expectedResponse = new()
            {
                Comments = new()
                {
                    new() { Comments = "Comments 1" },
                    new() { Comments = "Comments 2" }
                },
                TotalCount = 20,
                Count = 2,
            };

            _commentServiceMock.Setup(service =>
                service.RetrieveFilteredPositiveCommentsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResponse);

            PaginatedCommentsByDateRequest request = new()
            {
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Page = 1,
                PageSize = 10
            };

            CommentItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredPositiveCommentsAsync(request);

            Assert.AreEqual(expectedResponse.TotalCount, actualResponse.Count);

            _commentServiceMock.Verify(service =>
                service.RetrieveFilteredPositiveCommentsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}