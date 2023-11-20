namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidCommentException : Xeption
    {
        public InvalidCommentException() { }

        public InvalidCommentException(string message) : base(message) { }

        public InvalidCommentException(Exception innerException)
            : base(message: "Invalid Comment error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class NullCommentResponseException : Xeption
    {
        public NullCommentResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedCommentDependencyException : Xeption
    {
        public FailedCommentDependencyException(Exception innerException)
            : base(message: "Failed Comment dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedCommentServiceException : Xeption
    {
        public FailedCommentServiceException(Exception innerException)
            : base(message: "Failed Comment service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsCommentException : Exception
    {
        public AlreadyExistsCommentException(Exception innerException)
            : base(message: "Already exists Comment error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentValidationException : Exception
    {
        public CommentValidationException() { }

        public CommentValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentDependencyException : Exception
    {
        public CommentDependencyException() { }

        public CommentDependencyException(Exception innerException)
            : base($"CommentDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentDependencyValidationException : Exception
    {
        public CommentDependencyValidationException() { }

        public CommentDependencyValidationException(Exception innerException)
            : base($"CommentDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class CommentServiceException : Exception
    {
        public CommentServiceException() { }

        public CommentServiceException(Exception innerException)
            : base($"Internal or unknown system failure (CommentServiceException): {innerException.Message}", innerException) { }
    }
}
