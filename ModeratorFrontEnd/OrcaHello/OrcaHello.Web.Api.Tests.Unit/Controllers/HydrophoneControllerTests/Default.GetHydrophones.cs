namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class HydrophonesControllerTests
    {
        [TestMethod]
        public void Default_GetHydrophones_Expect_HydrophoneListResponse()
        {
            HydrophoneListResponse response = new()
            {
                Hydrophones = new List<Hydrophone>
                {
                    new()
                    {
                        Name = "Test"
                    }
                },
                Count = 1
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveHydrophoneLocations())
            .Returns(response);

            ActionResult<HydrophoneListResponse> actionResult =
                _controller.GetHydrophones();

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Hydrophones.Count,
                ((HydrophoneListResponse)contentResult.Value!).Hydrophones.Count);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveHydrophoneLocations(),
                Times.Once);
        }
    }
}
