namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class CommentServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new CommentServiceWrapper();

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            int invalidPage = -1;
            int validPageSize = 10;

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidatePagination(invalidPage, validPageSize));

            int validPage = 1;
            int invalidPageSize = -1;

            Assert.ThrowsException<InvalidCommentException>(() =>
                wrapper.ValidatePagination(validPage, invalidPageSize));

            CommentListResponse? invalidResponse = null;

            Assert.ThrowsException<NullCommentResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
