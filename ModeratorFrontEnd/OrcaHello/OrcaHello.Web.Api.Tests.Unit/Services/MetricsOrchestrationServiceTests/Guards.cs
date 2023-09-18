using OrcaHello.Web.Api.Models;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetricsOrchestrationServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new MetricsOrchestrationServiceWrapper();

            DateTime? invalidDate = DateTime.MinValue;

            Assert.ThrowsException<InvalidMetricOrchestrationException>(() =>
                wrapper.Validate(invalidDate, nameof(invalidDate)));

            DateTime? nullDate = null;

            Assert.ThrowsException<InvalidMetricOrchestrationException>(() =>
                wrapper.Validate(nullDate, nameof(nullDate)));
        }
    }
}
