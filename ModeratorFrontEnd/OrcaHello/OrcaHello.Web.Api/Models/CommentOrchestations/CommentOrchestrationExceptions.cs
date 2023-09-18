namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidCommentOrchestrationException : Exception
    {
        public InvalidCommentOrchestrationException() { }

        public InvalidCommentOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentOrchestrationValidationException : Exception
    {
        public CommentOrchestrationValidationException() { }

        public CommentOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentOrchestrationDependencyException : Exception
    {
        public CommentOrchestrationDependencyException() { }

        public CommentOrchestrationDependencyException(Exception innerException)
            : base($"CommentOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentOrchestrationDependencyValidationException : Exception
    {
        public CommentOrchestrationDependencyValidationException() { }

        public CommentOrchestrationDependencyValidationException(Exception innerException)
            : base($"CommentOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentOrchestrationServiceException : Exception
    {
        public CommentOrchestrationServiceException() { }

        public CommentOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (CommentOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}