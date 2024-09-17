namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidDetectionException : Xeption
    {
        public InvalidDetectionException() { }

        public InvalidDetectionException(string message) : base(message) { }

        public InvalidDetectionException(Exception innerException)
            : base(message: "Invalid Detection error occurred, please fix the errors and try again.",
              innerException,
              innerException.Data)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class NullDetectionRequestException : Xeption
    {
        public NullDetectionRequestException()
            : base(message: "Request is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullDetectionResponseException : Xeption
    {
        public NullDetectionResponseException()
            : base(message: "API call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedDetectionDependencyException : Xeption
    {
        public FailedDetectionDependencyException(Exception innerException)
            : base(message: "Failed Detection dependency error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class FailedDetectionServiceException : Xeption
    {
        public FailedDetectionServiceException(Exception innerException)
            : base(message: "Failed Detection service error occurred, please contact support.", innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class AlreadyExistsDetectionException : Exception
    {
        public AlreadyExistsDetectionException(Exception innerException)
            : base(message: "Already exists Detection error occurred, please try again.",
                  innerException)
        { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionValidationException : Exception
    {
        public DetectionValidationException() { }

        public DetectionValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionDependencyException : Exception
    {
        public DetectionDependencyException() { }

        public DetectionDependencyException(Exception innerException)
            : base($"DetectionDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionDependencyValidationException : Exception
    {
        public DetectionDependencyValidationException() { }

        public DetectionDependencyValidationException(Exception innerException)
            : base($"DetectionDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DetectionServiceException : Exception
    {
        public DetectionServiceException() { }

        public DetectionServiceException(Exception innerException)
            : base($"Internal or unknown system failure (DetectionServiceException): {innerException.Message}", innerException) { }
    }
}
