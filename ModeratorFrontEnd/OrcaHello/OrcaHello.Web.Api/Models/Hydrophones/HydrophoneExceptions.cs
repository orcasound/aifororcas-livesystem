namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidHydrophoneException : Exception
    {
        public InvalidHydrophoneException() { }

        public InvalidHydrophoneException(string message) : base(message) { }
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
