namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidMetricOrchestrationException : Exception
    {
        public InvalidMetricOrchestrationException() { }

        public InvalidMetricOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricOrchestrationValidationException : Exception
    {
        public MetricOrchestrationValidationException() { }

        public MetricOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricOrchestrationDependencyException : Exception
    {
        public MetricOrchestrationDependencyException() { }

        public MetricOrchestrationDependencyException(Exception innerException)
            : base($"MetricOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricOrchestrationDependencyValidationException : Exception
    {
        public MetricOrchestrationDependencyValidationException() { }

        public MetricOrchestrationDependencyValidationException(Exception innerException)
            : base($"MetricOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricOrchestrationServiceException : Exception
    {
        public MetricOrchestrationServiceException() { }

        public MetricOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (MetricOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}
