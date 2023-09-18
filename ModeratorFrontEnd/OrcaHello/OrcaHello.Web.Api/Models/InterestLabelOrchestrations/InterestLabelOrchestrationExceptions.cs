namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidInterestLabelOrchestrationException : Exception
    {
        public InvalidInterestLabelOrchestrationException() { }

        public InvalidInterestLabelOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class InterestLabelOrchestrationValidationException : Exception
    {
        public InterestLabelOrchestrationValidationException() { }

        public InterestLabelOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class InterestLabelOrchestrationDependencyException : Exception
    {
        public InterestLabelOrchestrationDependencyException() { }

        public InterestLabelOrchestrationDependencyException(Exception innerException)
            : base($"InterestLabelOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class InterestLabelOrchestrationDependencyValidationException : Exception
    {
        public InterestLabelOrchestrationDependencyValidationException() { }

        public InterestLabelOrchestrationDependencyValidationException(Exception innerException)
            : base($"InterestLabelOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class InterestLabelOrchestrationServiceException : Exception
    {
        public InterestLabelOrchestrationServiceException() { }

        public InterestLabelOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (InterestLabelOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}
