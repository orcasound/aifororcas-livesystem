namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class InvalidModeratorOrchestrationException : Exception
    {
        public InvalidModeratorOrchestrationException() { }

        public InvalidModeratorOrchestrationException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorOrchestrationValidationException : Exception
    {
        public ModeratorOrchestrationValidationException() { }

        public ModeratorOrchestrationValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorOrchestrationDependencyException : Exception
    {
        public ModeratorOrchestrationDependencyException() { }

        public ModeratorOrchestrationDependencyException(Exception innerException)
            : base($"ModeratorOrchestrationDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorOrchestrationDependencyValidationException : Exception
    {
        public ModeratorOrchestrationDependencyValidationException() { }

        public ModeratorOrchestrationDependencyValidationException(Exception innerException)
            : base($"ModeratorOrchestrationDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class ModeratorOrchestrationServiceException : Exception
    {
        public ModeratorOrchestrationServiceException() { }

        public ModeratorOrchestrationServiceException(Exception innerException)
            : base($"Internal or unknown system failure (ModeratorOrchestrationServiceException): {innerException.Message}", innerException) { }
    }
}