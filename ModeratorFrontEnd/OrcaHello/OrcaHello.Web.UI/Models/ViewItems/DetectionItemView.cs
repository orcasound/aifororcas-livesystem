namespace OrcaHello.Web.UI.Models
{
    public class DetectionItemView
    {
        public string Id { get; set; }
        public string AudioUri { get; set; }
        public string SpectrogramUri { get; set; }
        public string State { get; set; }
        public string LocationName { get; set; }
        public decimal Confidence { get; set; }
        public LocationItemView Location { get; set; }
        public List<AnnotationItemView> Annotations { get; set; }
        public DateTime Timestamp { get; set; }
        public string Comments { get; set; }
        public string Moderator { get; set; }
        public DateTime? Moderated { get; set; }
        public string Tags { get; set; }
        public string InterestLabel { get; set; }
        public bool IsCurrentlyPlaying { get; set; }
        public string AverageConfidence { get => $"{Confidence.ToString("00.##")}% average confidence"; }
        public string SmallConfidence { get => $"{Confidence.ToString("F2")}%"; }
        public string DetectionCount { get => $"{Annotations.Count} detections"; }
        
        public List<string> TagsList
        {
            get
            {
                if (string.IsNullOrWhiteSpace(Tags))
                    return new List<string>();

                return Tags.Split(',').ToList();
            }
            set
            {
                if(value != null && value.Count > 0)
                    Tags = string.Join(',', value);
            }
        }

        public string EnteredTags
        {
            get
            {
                return string.Empty;
            }
            set
            {
                if (!string.IsNullOrWhiteSpace(value))
                {
                    var enteredTagsList = value
                        .Split(new char[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries)
                        .ToList()
                        .Select(s => s.Trim());

                    var workingTagsList = !string.IsNullOrWhiteSpace(Tags) ?
                        Tags.Split(',').ToList() : new List<string>();

                    foreach(var tag in enteredTagsList)
                    {
                        if (!workingTagsList.Contains(tag))
                            workingTagsList.Add(tag);
                    }

                    if(workingTagsList.Count > 0)
                        Tags = string.Join(",", workingTagsList);
                }
            }
        }

        public static Func<Detection, DetectionItemView> AsDetectionItemView =>
             detection => new DetectionItemView
             {
                 Id = detection.Id,
                 LocationName = detection.LocationName,
                 Timestamp = detection.Timestamp,
                 AudioUri = detection.AudioUri,
                 SpectrogramUri = detection.SpectrogramUri,
                 Confidence = detection.Confidence,
                 State = detection.State,
                 Location = new()
                 {
                     Name = detection.Location.Name,
                     Longitude = detection.Location.Longitude,
                     Latitude = detection.Location.Latitude
                 },
                 Comments = detection.Comments,
                 Moderator = detection.Moderator,
                 Moderated = detection.Moderated,
                 Tags = String.Join(", ", detection.Tags),
                 Annotations = detection.Annotations.Select(AsAnnotationItemView).ToList(),
                 InterestLabel = detection.InterestLabel
             };

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