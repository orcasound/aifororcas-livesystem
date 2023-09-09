namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class HydrophonesControllerTests
    {

        [TestMethod]
        public void TryCatch_GetHydrophones_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveHydrophoneLocations())

                .Throws(new HydrophoneOrchestrationValidationException(new Exception()))
                .Throws(new HydrophoneOrchestrationDependencyValidationException(new Exception()))

                .Throws(new HydrophoneOrchestrationDependencyException(new Exception()))
                .Throws(new HydrophoneOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            ExecuteRetrieveHydrophones(2, StatusCodes.Status400BadRequest);
            ExecuteRetrieveHydrophones(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveHydrophoneLocations(),
                    Times.Exactly(5));

        }

        private void ExecuteRetrieveHydrophones(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<HydrophoneListResponse> actionResult =
                    _controller.GetHydrophones();

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
