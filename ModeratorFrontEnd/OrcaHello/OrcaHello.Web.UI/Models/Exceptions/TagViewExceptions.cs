namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullTagViewResponseException : Xeption
    {
        public NullTagViewResponseException()
            : base(message: "Orchestration service call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullTagViewRequestException : Xeption
    {
        public NullTagViewRequestException()
            : base(message: "Request is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullTagViewException : Xeption
    {
        public NullTagViewException()
            : base(message: "TagItemView is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidTagViewException : Xeption
    {
        public InvalidTagViewException() { }

        public InvalidTagViewException(string message) : base(message) { }

        public InvalidTagViewException(Exception innerException)
            : base(message: "Invalid TagView error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class TagViewValidationException : Exception
    {
        public TagViewValidationException() { }

        public TagViewValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagViewDependencyValidationException : Exception
    {
        public TagViewDependencyValidationException() { }

        public TagViewDependencyValidationException(Exception innerException)
            : base($"TagViewDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagViewDependencyException : Exception
    {
        public TagViewDependencyException() { }

        public TagViewDependencyException(Exception innerException)
            : base($"TagViewDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagViewServiceException : Exception
    {
        public TagViewServiceException() { }

        public TagViewServiceException(Exception innerException)
            : base($"Internal or unknown system failure (TagViewServiceException): {innerException.Message}", innerException) { }
    }
}
