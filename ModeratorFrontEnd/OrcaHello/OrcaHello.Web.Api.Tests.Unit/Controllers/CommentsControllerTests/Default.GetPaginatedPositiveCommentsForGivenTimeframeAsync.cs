namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class CommentsControllerTests
    {
        [TestMethod]
        public async Task Default_GetGetPaginatedPositiveCommentsForGivenTimeframeAsync_Expect_DetectionListResponse()
        {
            CommentListResponse response = new()
            {
                Count = 2,
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Comments = new List<Comment> { new() }
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrievePositiveCommentsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(response);

            ActionResult<CommentListResponse> actionResult =
                await _controller.GetPaginatedPositiveCommentsForGivenTimeframeAsync(DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Comments.Count,
                ((CommentListResponse)contentResult.Value!).Comments.Count);

            _orchestrationServiceMock.Verify(service =>
               service.RetrievePositiveCommentsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}
