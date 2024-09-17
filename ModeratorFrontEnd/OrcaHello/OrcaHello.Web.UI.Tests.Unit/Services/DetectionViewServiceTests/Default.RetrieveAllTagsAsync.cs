using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllTagsAsync()
        {
            List<string> expectedResponse = new()
            {
                "Tag 1"
            };

            _tagServiceMock.Setup(service =>
                service.RetrieveAllTagsAsync())
                .ReturnsAsync(expectedResponse);

            List<string> actualResponse =
                await _viewService.RetrieveAllTagsAsync();

            Assert.AreEqual(expectedResponse.Count(), actualResponse.Count());

            _tagServiceMock.Verify(service =>
                service.RetrieveAllTagsAsync(),
                Times.Once);
        }
    }
}
