using Microsoft.AspNetCore.Http;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class CommentServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredPositiveCommentsAsync()
        {
            CommentListResponse expectedResponse = new()
            {
                Comments = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Comments = "Test comment." }
                }
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredPositiveCommentsAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            var result = await _service.RetrieveFilteredPositiveCommentsAsync(DateTime.UtcNow.AddDays(-14), DateTime.UtcNow, 1, 10);

            Assert.AreEqual(expectedResponse.Comments.Count(), result.Comments.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredPositiveCommentsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}
