namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidDetectionViewException : Exception
    {
        public InvalidDetectionViewException() { }

        public InvalidDetectionViewException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class NullDetectionViewResponseException : Xeption
    {
        public NullDetectionViewResponseException(string name)
            : base(message: $"Service call returned a null '{name}' response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullDetectionViewRequestException : Xeption
    {
        public NullDetectionViewRequestException(string name)
            : base(message: $"Request '{name}' calling method is null.") { }
    }


    [ExcludeFromCodeCoverage]
    public class DetectionViewValidationException : Exception
    {
        public DetectionViewValidationException() { }

        public DetectionViewValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionViewDependencyValidationException : Exception
    {
        public DetectionViewDependencyValidationException() { }

        public DetectionViewDependencyValidationException(Exception innerException)
            : base($"DetectionViewDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionViewDependencyException : Exception
    {
        public DetectionViewDependencyException() { }

        public DetectionViewDependencyException(Exception innerException)
            : base($"DetectionViewDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionViewServiceException : Exception
    {
        public DetectionViewServiceException() { }

        public DetectionViewServiceException(Exception innerException)
            : base($"Internal or unknown system failure (DetectionViewServiceException): {innerException.Message}", innerException) { }
    }
}
