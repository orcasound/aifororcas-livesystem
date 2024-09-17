namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullTagResponseException : Xeption
    {
        public NullTagResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullTagRequestException : Xeption
    {
        public NullTagRequestException()
            : base(message: "Request is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidTagException : Xeption
    {
        public InvalidTagException() { }

        public InvalidTagException(string message) : base(message) { }

        public InvalidTagException(Exception innerException)
            : base(message: "Invalid Tag error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedTagDependencyException : Xeption
    {
        public FailedTagDependencyException(Exception innerException)
            : base(message: "Failed Tag dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedTagServiceException : Xeption
    {
        public FailedTagServiceException(Exception innerException)
            : base(message: "Failed Tag service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsTagException : Exception
    {
        public AlreadyExistsTagException(Exception innerException)
            : base(message: "Already exists Tag error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class TagValidationException : Exception
    {
        public TagValidationException() { }

        public TagValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagDependencyException : Exception
    {
        public TagDependencyException() { }

        public TagDependencyException(Exception innerException)
            : base($"TagDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagDependencyValidationException : Exception
    {
        public TagDependencyValidationException() { }

        public TagDependencyValidationException(Exception innerException)
            : base($"TagDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagServiceException : Exception
    {
        public TagServiceException() { }

        public TagServiceException(Exception innerException)
            : base($"Internal or unknown system failure (TagServiceException): {innerException.Message}", innerException) { }
    }

}