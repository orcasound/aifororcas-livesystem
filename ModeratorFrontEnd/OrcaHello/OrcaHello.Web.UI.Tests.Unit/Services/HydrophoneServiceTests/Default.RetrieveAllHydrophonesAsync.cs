using OrcaHello.Web.UI.Pages.Explore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class HydrophoneServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllHydrophonesAsync()
        {
            HydrophoneListResponse expectedResponse = new()
            {
                Hydrophones = new()
                {
                    new() { Name = "Sample" }
                },
                Count = 1
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetAllHydrophonesAsync())
                .ReturnsAsync(expectedResponse);

            var result = await _service.RetrieveAllHydrophonesAsync();

            Assert.AreEqual(expectedResponse.Hydrophones.Count(), result.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetAllHydrophonesAsync(),
                Times.Once);
        }
    }
}
