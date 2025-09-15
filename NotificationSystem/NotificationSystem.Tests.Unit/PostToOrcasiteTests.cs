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

        /// <summary>
        /// This test verifies that the mock HTTP client is actually called when processing documents.
        /// It uses the verification capabilities of MockHttpMessageHandler to ensure HTTP requests are made.
        /// </summary>
        [Fact]
        public async Task PostToOrcasite_ProcessDocumentsAsync_VerifiesHttpCallsMade()
        {
            var mockOrcasiteLogger = new Mock<ILogger<OrcasiteHelper>>();
            var (orcasiteHelper, mockHttp) = OrcasiteTestHelper.GetMockOrcasiteHelperWithVerification(mockOrcasiteLogger.Object);
            List<JsonElement> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Process it like the Azure function would.
            var mockFunctionLogger = new Mock<ILogger<PostToOrcasite>>();
            var postToOrcasite = new PostToOrcasite(orcasiteHelper, mockFunctionLogger.Object);
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents);
            
            // Assert that the operation succeeded
            Assert.True(ok, "PostToOrcasite.ProcessDocumentsAsync failed");
            
            // Verify that all expected HTTP calls were made
            mockHttp.VerifyNoOutstandingExpectation();
        }
    }
}
