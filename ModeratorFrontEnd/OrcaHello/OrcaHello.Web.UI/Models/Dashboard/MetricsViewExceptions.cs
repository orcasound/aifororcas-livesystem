namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidMetricsViewException : Exception
    {
        public InvalidMetricsViewException() { }

        public InvalidMetricsViewException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsViewValidationException : Exception
    {
        public MetricsViewValidationException() { }

        public MetricsViewValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsViewDependencyValidationException : Exception
    {
        public MetricsViewDependencyValidationException() { }

        public MetricsViewDependencyValidationException(Exception innerException)
            : base($"MetricsViewDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsViewDependencyException : Exception
    {
        public MetricsViewDependencyException() { }

        public MetricsViewDependencyException(Exception innerException)
            : base($"MetricsViewDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsViewServiceException : Exception
    {
        public MetricsViewServiceException() { }

        public MetricsViewServiceException(Exception innerException)
            : base($"Internal or unknown system failure (MetricsViewServiceException): {innerException.Message}", innerException) { }
    }

}
