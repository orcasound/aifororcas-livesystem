namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class CommentOrchestrationTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new CommentOrchestrationServiceWrapper();

            DateTime? invalidDate = DateTime.MinValue;

            Assert.ThrowsException<InvalidCommentOrchestrationException>(() =>
                wrapper.Validate(invalidDate, nameof(invalidDate)));

            DateTime? nullDate = null;

            Assert.ThrowsException<InvalidCommentOrchestrationException>(() =>
                wrapper.Validate(nullDate, nameof(nullDate)));

            int invalidPage = 0;

            Assert.ThrowsException<InvalidCommentOrchestrationException>(() =>
                wrapper.ValidatePage(invalidPage));

            int invalidPageSize = 0;
            Assert.ThrowsException<InvalidCommentOrchestrationException>(() =>
                wrapper.ValidatePageSize(invalidPageSize));
        }
    }
}
