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
        /// It also verifies that the mock HTTP client is actually called when processing documents.
        /// </summary>
        [Fact]
        public async Task PostToOrcasite_ProcessDocumentsAsync()
        {
            var mockOrcasiteLogger = new Mock<ILogger<OrcasiteHelper>>();
            var container = OrcasiteTestHelper.GetMockOrcasiteHelperWithRequestVerification(mockOrcasiteLogger.Object);
            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Process it like the Azure function would.
            var mockFunctionLogger = new Mock<ILogger<PostToOrcasite>>();
            var postToOrcasite = new PostToOrcasite(container.Helper, mockFunctionLogger.Object);
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents);
            
            // Assert that the operation succeeded.
            Assert.True(ok, "PostToOrcasite.ProcessDocumentsAsync failed");
            
            // Verify that the expected number of HTTP calls were made (1 GET feeds + 1 POST detection).
            Assert.Equal(1, container.MockHttp.GetMatchCount(container.GetFeedsRequest));
            Assert.Equal(1, container.MockHttp.GetMatchCount(container.PostDetectionRequest));
        }
    }
}
