namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidHydrophoneViewException : Exception
    {
        public InvalidHydrophoneViewException() { }

        public InvalidHydrophoneViewException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class NullHydrophoneViewResponseException : Xeption
    {
        public NullHydrophoneViewResponseException(string name)
            : base(message: $"Service call returned a null '{name}' response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneViewValidationException : Exception
    {
        public HydrophoneViewValidationException() { }

        public HydrophoneViewValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneViewDependencyValidationException : Exception
    {
        public HydrophoneViewDependencyValidationException() { }

        public HydrophoneViewDependencyValidationException(Exception innerException)
            : base($"HydrophoneViewDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneViewDependencyException : Exception
    {
        public HydrophoneViewDependencyException() { }

        public HydrophoneViewDependencyException(Exception innerException)
            : base($"HydrophoneViewDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneViewServiceException : Exception
    {
        public HydrophoneViewServiceException() { }

        public HydrophoneViewServiceException(Exception innerException)
            : base($"Internal or unknown system failure (HydrophoneViewServiceException): {innerException.Message}", innerException) { }
    }
}
