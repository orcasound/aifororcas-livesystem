namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class NullDashboardViewRequestException : Xeption
    {
        public NullDashboardViewRequestException()
            : base(message: "Request is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullDashboardViewResponseException : Xeption
    {
        public NullDashboardViewResponseException()
            : base(message: "Foundation Service call returned a null response.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidDashboardViewException : Exception
    {
        public InvalidDashboardViewException() { }

        public InvalidDashboardViewException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class DashboardViewValidationException : Exception
    {
        public DashboardViewValidationException() { }

        public DashboardViewValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DashboardViewDependencyValidationException : Exception
    {
        public DashboardViewDependencyValidationException() { }

        public DashboardViewDependencyValidationException(Exception innerException)
            : base($"DashboardViewDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DashboardViewDependencyException : Exception
    {
        public DashboardViewDependencyException() { }

        public DashboardViewDependencyException(Exception innerException)
            : base($"DashboardViewDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class DashboardViewServiceException : Exception
    {
        public DashboardViewServiceException() { }

        public DashboardViewServiceException(Exception innerException)
            : base($"Internal or unknown system failure (DashboardViewServiceException): {innerException.Message}", innerException) { }
    }

}
