namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task Default_GetModeratorsAsync_Expect_DetectionListResponse()
        {
            ModeratorListResponse response = new()
            {
                Moderators = new List<string> { "Moderator 1", "Moderator 2" },
                Count = 2
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveModeratorsAsync())
                .ReturnsAsync(response);

            ActionResult<ModeratorListResponse> actionResult =
                await _controller.GetModeratorsAsync();

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Moderators.Count,
                ((ModeratorListResponse)contentResult.Value!).Moderators.Count);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveModeratorsAsync(),
                Times.Once);
        }
    }
}
