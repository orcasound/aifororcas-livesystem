using OrcaHello.Web.Api.Services;
using OrcaHello.Web.Shared.Models.Comments;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class CommentOrchestrationServiceWrapper : CommentOrchestrationService
    {
        public new void Validate(DateTime? date, string propertyName) =>
            base.Validate(date, propertyName);

        public new void ValidatePage(int page) =>
            base.ValidatePage(page);

        public new void ValidatePageSize(int pageSize) =>
            base.ValidatePageSize(pageSize);

        public new ValueTask<CommentListResponse> TryCatch(ReturningCommentListResponseFunction returningCommentListResponseFunction) =>
            base.TryCatch(returningCommentListResponseFunction);
    }
}
