namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidDetectionOrchestrationException : Exception
    {
        public InvalidDetectionOrchestrationException() { }

        public InvalidDetectionOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionOrchestrationValidationException : Exception
    {
        public DetectionOrchestrationValidationException() { }

        public DetectionOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionOrchestrationDependencyException : Exception
    {
        public DetectionOrchestrationDependencyException() { }

        public DetectionOrchestrationDependencyException(Exception innerException)
            : base($"DetectionOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionOrchestrationDependencyValidationException : Exception
    {
        public DetectionOrchestrationDependencyValidationException() { }

        public DetectionOrchestrationDependencyValidationException(Exception innerException)
            : base($"DetectionOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionOrchestrationServiceException : Exception
    {
        public DetectionOrchestrationServiceException() { }

        public DetectionOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (DetectionOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}
