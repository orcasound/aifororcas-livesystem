namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullModeratorResponseException : Xeption
    {
        public NullModeratorResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidModeratorException : Xeption
    {
        public InvalidModeratorException() { }

        public InvalidModeratorException(string message) : base(message) { }

        public InvalidModeratorException(Exception innerException)
            : base(message: "Invalid Moderator error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedModeratorDependencyException : Xeption
    {
        public FailedModeratorDependencyException(Exception innerException)
            : base(message: "Failed Moderator dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedModeratorServiceException : Xeption
    {
        public FailedModeratorServiceException(Exception innerException)
            : base(message: "Failed Moderator service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsModeratorException : Exception
    {
        public AlreadyExistsModeratorException(Exception innerException)
            : base(message: "Already exists Moderator error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorValidationException : Exception
    {
        public ModeratorValidationException() { }

        public ModeratorValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorDependencyException : Exception
    {
        public ModeratorDependencyException() { }

        public ModeratorDependencyException(Exception innerException)
            : base($"ModeratorDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorDependencyValidationException : Exception
    {
        public ModeratorDependencyValidationException() { }

        public ModeratorDependencyValidationException(Exception innerException)
            : base($"ModeratorDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorServiceException : Exception
    {
        public ModeratorServiceException() { }

        public ModeratorServiceException(Exception innerException)
            : base($"Internal or unknown system failure (ModeratorServiceException): {innerException.Message}", innerException) { }
    }
}
