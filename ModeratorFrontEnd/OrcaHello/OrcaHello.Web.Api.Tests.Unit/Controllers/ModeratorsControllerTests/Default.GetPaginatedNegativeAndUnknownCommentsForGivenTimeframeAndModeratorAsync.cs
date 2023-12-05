namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task Default_GetGetPaginatedNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync_Expect_DetectionListResponse()
        {
            CommentListForModeratorResponse response = new()
            {
                Count = 2,
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Comments = new List<Comment> { new() },
                Moderator = "Moderator"
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(response);

            ActionResult<CommentListForModeratorResponse> actionResult =
                await _controller.GetPaginatedNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Comments.Count,
                ((CommentListForModeratorResponse)contentResult.Value!).Comments.Count);

            _orchestrationServiceMock.Verify(service =>
               service.RetrieveNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}
