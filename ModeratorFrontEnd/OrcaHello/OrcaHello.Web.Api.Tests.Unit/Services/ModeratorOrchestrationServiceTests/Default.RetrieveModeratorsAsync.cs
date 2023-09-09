using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Shared.Models.Moderators;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class ModeratorOrchestrationServiceTests
    {
        [TestMethod]
        public async Task Default_RetrieveModeratorsAsync_Expect()
        {
            QueryableModerators expectedResult = new QueryableModerators
            {
                QueryableRecords = (new List<string>
                {
                    "Moderator 1",
                    "Moderator 2"
                }).AsQueryable(),
                TotalCount = 2
            };

            _metadataServiceMock.Setup(service =>
                service.RetrieveModeratorsAsync())
                .ReturnsAsync(expectedResult);

            ModeratorListResponse result = await _orchestrationService.RetrieveModeratorsAsync();

            Assert.AreEqual(expectedResult.QueryableRecords.Count(), result.Moderators.Count());

            _metadataServiceMock.Verify(service =>
                service.RetrieveModeratorsAsync(),
                    Times.Once);
        }
    }
}