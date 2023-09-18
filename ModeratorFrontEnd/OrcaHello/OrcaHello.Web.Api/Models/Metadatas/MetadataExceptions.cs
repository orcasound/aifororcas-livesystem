namespace OrcaHello.Web.Api.Models
{

    [ExcludeFromCodeCoverage]
    public class NotFoundMetadataException : Exception
    {
        public NotFoundMetadataException(string id)
            : base(message: $"Couldn't find metadata with id: {id}.") { }
    }

    [ExcludeFromCodeCoverage]
    public class NullMetadataException : Exception
    {
        public NullMetadataException() : base(message: "The metadata is null.") { }
    }

    [ExcludeFromCodeCoverage]
    public class InvalidMetadataException : Exception
    {
        public InvalidMetadataException() { }

        public InvalidMetadataException(string message) : base(message) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetadataValidationException : Exception
    {
        public MetadataValidationException() { }

        public MetadataValidationException(Exception innerException)
            : base($"Invalid input: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetadataDependencyException : Exception
    {
        public MetadataDependencyException() { }

        public MetadataDependencyException(Exception innerException)
            : base($"MetadataDependency exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetadataDependencyValidationException : Exception
    {
        public MetadataDependencyValidationException() { }

        public MetadataDependencyValidationException(Exception innerException)
            : base($"MetadataDependencyValidation exception: {innerException.Message}", innerException) { }
    }

    [ExcludeFromCodeCoverage]
    public class MetadataServiceException : Exception
    {
        public MetadataServiceException() { }

        public MetadataServiceException(Exception innerException)
            : base($"Internal or unknown system failure (MetadataServiceException): {innerException.Message}", innerException) { }
    }
}
