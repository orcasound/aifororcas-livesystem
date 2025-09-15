using Microsoft.Extensions.Logging;
using Moq;
using NotificationSystem.Models;
using NotificationSystem.Tests.Common;
using System.Text.Json;

namespace NotificationSystem.Tests.Unit
{

    public class PostToOrcasiteTests
    {
        /// <summary>
        /// This test mocks the Orcasite API and verifies that a detection can be posted successfully.
        /// </summary>
        /// <returns></returns>
        [Fact]
        public async Task PostToOrcasite_ProcessDocumentsAsync()
        {
            var mockOrcasiteLogger = new Mock<ILogger<OrcasiteHelper>>();
            OrcasiteHelper orcasiteHelper = OrcasiteTestHelper.GetMockOrcasiteHelper(mockOrcasiteLogger.Object);
            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Process it like the Azure function would.
            var mockFunctionLogger = new Mock<ILogger<PostToOrcasite>>();
            var postToOrcasite = new PostToOrcasite(orcasiteHelper, mockFunctionLogger.Object);
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents);
            Assert.True(ok, "PostToOrcasite.ProcessDocumentsAsync failed");
        }
    }
}
