namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidHydrophoneOrchestrationException : Exception
    {
        public InvalidHydrophoneOrchestrationException() { }

        public InvalidHydrophoneOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationValidationException : Exception
    {
        public HydrophoneOrchestrationValidationException() { }

        public HydrophoneOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationDependencyException : Exception
    {
        public HydrophoneOrchestrationDependencyException() { }

        public HydrophoneOrchestrationDependencyException(Exception innerException)
            : base($"HydrophoneOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationDependencyValidationException : Exception
    {
        public HydrophoneOrchestrationDependencyValidationException() { }

        public HydrophoneOrchestrationDependencyValidationException(Exception innerException)
            : base($"HydrophoneOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationServiceException : Exception
    {
        public HydrophoneOrchestrationServiceException() { }

        public HydrophoneOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (HydrophoneOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}