namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullHydrophoneResponseException : Xeption
    {
        public NullHydrophoneResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidHydrophoneException : Xeption
    {
        public InvalidHydrophoneException() { }

        public InvalidHydrophoneException(string message) : base(message) { }

        public InvalidHydrophoneException(Exception innerException)
            : base(message: "Invalid hydrophone error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedHydrophoneDependencyException : Xeption
    {
        public FailedHydrophoneDependencyException(Exception innerException)
            : base(message: "Failed hydrophone dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedHydrophoneServiceException : Xeption
    {
        public FailedHydrophoneServiceException(Exception innerException)
            : base(message: "Failed hydrophone service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsHydrophoneException : Exception
    {
        public AlreadyExistsHydrophoneException(Exception innerException)
            : base(message: "Already exists hydrophone error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneValidationException : Exception
    {
        public HydrophoneValidationException() { }

        public HydrophoneValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneDependencyException : Exception
    {
        public HydrophoneDependencyException() { }

        public HydrophoneDependencyException(Exception innerException)
            : base($"HydrophoneDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneDependencyValidationException : Exception
    {
        public HydrophoneDependencyValidationException() { }

        public HydrophoneDependencyValidationException(Exception innerException)
            : base($"HydrophoneDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneServiceException : Exception
    {
        public HydrophoneServiceException() { }

        public HydrophoneServiceException(Exception innerException)
            : base($"Internal or unknown system failure (HydrophoneServiceException): {innerException.Message}", innerException) { }
    }
}
