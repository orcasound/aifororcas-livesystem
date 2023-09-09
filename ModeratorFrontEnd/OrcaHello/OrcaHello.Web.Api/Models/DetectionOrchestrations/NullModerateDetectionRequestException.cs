namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class NullModerateDetectionRequestException : Exception
    {
        public NullModerateDetectionRequestException() : base(message: "The request is null.") { }
    }
}
