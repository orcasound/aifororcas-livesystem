namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredTagsForModeratorAsync()
        {
            TagListForModeratorResponse expectedResponse = new()
            {
                Tags = new() { "Tag 1", "Tag 2" },
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Count = 2,
                Moderator = "Moderator"
            };

            _moderatorServiceMock.Setup(service =>
                service.GetFilteredTagsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                    .ReturnsAsync(expectedResponse);

            TagsByDateRequest request = new()
            {
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
            };

            List<string> actualResponse =
                await _viewService.RetrieveFilteredTagsForModeratorAsync("Moderator", request);

            Assert.AreEqual(expectedResponse.Count, actualResponse.Count);

            _moderatorServiceMock.Verify(service =>
                service.GetFilteredTagsForModeratorAsync(It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}