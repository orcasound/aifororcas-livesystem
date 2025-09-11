using Microsoft.Extensions.Logging;
using Moq;
using NotificationSystem.Models;
using NotificationSystem.Tests.Common;

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
            var mockLogger = new Mock<ILogger<PostToOrcasite>>();
            OrcasiteHelper orcasiteHelper = OrcasiteTestHelper.GetMockOrcasiteHelper(mockLogger.Object);
            List<dynamic> documents = OrcasiteTestHelper.GetSampleOrcaHelloDetections();

            // Process it like the Azure function would.
            var postToOrcasite = new PostToOrcasite(orcasiteHelper, mockLogger.Object);
            bool ok = await postToOrcasite.ProcessDocumentsAsync(documents);
            Assert.True(ok, "PostToOrcasite.ProcessDocumentsAsync failed");
        }
    }
}
