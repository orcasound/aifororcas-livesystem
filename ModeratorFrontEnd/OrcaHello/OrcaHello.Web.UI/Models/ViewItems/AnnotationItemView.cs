namespace OrcaHello.Web.UI.Models
{
    // This class represents a view model for an annotation item, which is used throughout the
    // various view services to display information about annotations.
    [ExcludeFromCodeCoverage]
    public class AnnotationItemView
    {
        // Unique identifier for the annotation.
        public int Id { get; set; } = 0;

        // Start time of the annotation in seconds.
        public decimal StartTime { get; set; } = decimal.Zero;

        // End time of the annotation in seconds.
        public decimal EndTime { get; set; } = decimal.Zero;

        // Confidence level associated with the annotation.
        public decimal Confidence { get; set; } = decimal.Zero;

        // Converts an Annotation object (from the API) into an AnnotationItemView using a delegate.
        public static Func<Annotation, AnnotationItemView> AsAnnotationItemView =>
            annotation => new AnnotationItemView
            {
                Id = annotation.Id,
                StartTime = annotation.StartTime,
                EndTime = annotation.EndTime,
                Confidence = annotation.Confidence
            };
    }
}
