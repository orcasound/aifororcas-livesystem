namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task Default_GetTagsForGivenTimePeriod_Expect_TagListResponse()
        {
            TagListForTimeframeResponse response = new()
            {
                Count = 2,
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Tags = new List<string> { "Tag1", "Tag2" }
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveTagsForGivenTimePeriodAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                .ReturnsAsync(response);

            ActionResult<TagListForTimeframeResponse> actionResult =
                await _controller.GetTagsForGivenTimeframeAsync(DateTime.Now, DateTime.Now.AddDays(1));

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Tags.Count,
                ((TagListResponse)contentResult.Value!).Tags.Count);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveTagsForGivenTimePeriodAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}
