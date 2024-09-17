namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task Default_GetTagsForGivenTimePeriodAndModerator_Expect_TagListResponse()
        {
            TagListForModeratorResponse response = new()
            {
                Count = 2,
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Tags = new List<string> { "Tag1", "Tag2" },
                Moderator = "Moderator"
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveTagsForGivenTimePeriodAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                .ReturnsAsync(response);

            ActionResult<TagListForModeratorResponse> actionResult =
                await _controller.GetTagsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1));

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Tags.Count,
                ((TagListResponse)contentResult.Value!).Tags.Count);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveTagsForGivenTimePeriodAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
