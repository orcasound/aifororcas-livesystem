namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetModeratorsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveModeratorsAsync())

                .Throws(new ModeratorOrchestrationValidationException(new Exception()))
                .Throws(new ModeratorOrchestrationDependencyValidationException(new Exception()))

                .Throws(new ModeratorOrchestrationDependencyException(new Exception()))
                .Throws(new ModeratorOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveModerators(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveModerators(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveModeratorsAsync(),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveModerators(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<ModeratorListResponse> actionResult =
                    await _controller.GetModeratorsAsync();

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }

    }
}
