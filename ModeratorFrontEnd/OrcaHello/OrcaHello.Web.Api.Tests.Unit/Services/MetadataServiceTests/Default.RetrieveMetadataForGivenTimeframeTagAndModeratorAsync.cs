using Azure;
using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveMetadataForGivenTimeframeTagAndModeratorAsync()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new List<Metadata>
                {
                    new()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetMetadataListByTimeframeTagAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveMetadataForGivenTimeframeTagAndModeratorAsync(fromDate, toDate, "moderator", "tag", 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count, result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetMetadataListByTimeframeTagAndModerator(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }
    }
}
