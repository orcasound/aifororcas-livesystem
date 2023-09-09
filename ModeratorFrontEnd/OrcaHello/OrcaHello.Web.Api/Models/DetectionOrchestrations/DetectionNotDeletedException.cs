namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class DetectionNotDeletedException : Exception
    {
        public DetectionNotDeletedException(string id)
            : base(message: $"Could not delete metadata with id: {id}.") { }
    }
}
