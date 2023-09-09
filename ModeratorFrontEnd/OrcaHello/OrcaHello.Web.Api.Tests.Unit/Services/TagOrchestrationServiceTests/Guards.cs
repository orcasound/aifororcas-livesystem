namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class TagOrchestrationServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new TagOrchestrationServiceWrapper();

            DateTime? invalidDate = DateTime.MinValue;

            Assert.ThrowsException<InvalidTagOrchestrationException>(() =>
                wrapper.Validate(invalidDate, nameof(invalidDate)));

            DateTime? nullDate = null;

            Assert.ThrowsException<InvalidTagOrchestrationException>(() =>
                wrapper.Validate(nullDate, nameof(nullDate)));

            string invalidProperty = string.Empty;

            Assert.ThrowsException<InvalidTagOrchestrationException>(() =>
                wrapper.Validate(invalidProperty, nameof(invalidProperty)));
        }
    }
}