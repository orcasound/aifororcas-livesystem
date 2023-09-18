namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidTagOrchestrationException : Exception
    {
        public InvalidTagOrchestrationException() { }

        public InvalidTagOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagOrchestrationValidationException : Exception
    {
        public TagOrchestrationValidationException() { }

        public TagOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagOrchestrationDependencyException : Exception
    {
        public TagOrchestrationDependencyException() { }

        public TagOrchestrationDependencyException(Exception innerException)
            : base($"TagOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagOrchestrationDependencyValidationException : Exception
    {
        public TagOrchestrationDependencyValidationException() { }

        public TagOrchestrationDependencyValidationException(Exception innerException)
            : base($"TagOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class TagOrchestrationServiceException : Exception
    {
        public TagOrchestrationServiceException() { }

        public TagOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (TagOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}
