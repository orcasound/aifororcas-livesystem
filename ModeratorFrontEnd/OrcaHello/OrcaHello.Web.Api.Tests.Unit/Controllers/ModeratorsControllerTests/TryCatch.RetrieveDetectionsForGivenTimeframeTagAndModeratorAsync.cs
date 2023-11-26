using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_RetrieveDetectionsForGivenTimeframeTagAndModeratorAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new ModeratorOrchestrationValidationException(new Exception()))
                .Throws(new ModeratorOrchestrationDependencyValidationException(new Exception()))

                .Throws(new ModeratorOrchestrationDependencyException(new Exception()))
                .Throws(new ModeratorOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));
        }

        private async Task ExecuteRetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<DetectionListForModeratorAndTagResponse> actionResult =
                    await _controller.GetPaginatedDetectionsForGivenTimeframeTagAndModeratorAsync("Moderator", "Tag", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
