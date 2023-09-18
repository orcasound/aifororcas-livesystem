namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class DetectionNotInsertedException : Exception
    {
        public DetectionNotInsertedException(string id)
            : base(message: $"Could not insert metadata with id: {id}.") { }
    }
}