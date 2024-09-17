namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullMetricsResponseException : Xeption
    {
        public NullMetricsResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidMetricsException : Xeption
    {
        public InvalidMetricsException() { }

        public InvalidMetricsException(string message) : base(message) { }

        public InvalidMetricsException(Exception innerException)
            : base(message: "Invalid Metrics error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedMetricsDependencyException : Xeption
    {
        public FailedMetricsDependencyException(Exception innerException)
            : base(message: "Failed Metrics dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedMetricsServiceException : Xeption
    {
        public FailedMetricsServiceException(Exception innerException)
            : base(message: "Failed Metrics service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsMetricsException : Exception
    {
        public AlreadyExistsMetricsException(Exception innerException)
            : base(message: "Already exists Metrics error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsValidationException : Exception
    {
        public MetricsValidationException() { }

        public MetricsValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsDependencyException : Exception
    {
        public MetricsDependencyException() { }

        public MetricsDependencyException(Exception innerException)
            : base($"MetricsDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsDependencyValidationException : Exception
    {
        public MetricsDependencyValidationException() { }

        public MetricsDependencyValidationException(Exception innerException)
            : base($"MetricsDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsServiceException : Exception
    {
        public MetricsServiceException() { }

        public MetricsServiceException(Exception innerException)
            : base($"Internal or unknown system failure (MetricsServiceException): {innerException.Message}", innerException) { }
    }

}
