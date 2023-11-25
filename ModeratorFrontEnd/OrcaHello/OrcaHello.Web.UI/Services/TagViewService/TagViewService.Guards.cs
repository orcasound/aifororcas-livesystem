namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="TagViewService"/> orchestration service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class TagViewService
    {
        // RULE: TagItemView must not be null and Tag property must have a value
        private static void ValidateTagItemView(TagItemView tagItemView)
        {
            if (tagItemView is null)
                throw new NullTagViewException();

            if (String.IsNullOrWhiteSpace(tagItemView.Tag))
                throw new InvalidTagViewException("The tag cannot be null, empty, or whitespace.");
        }

        // RULE: ReplaceTagRequest must not be null and both tag properties must have a value
        private static void ValidateReplaceTagRequest(ReplaceTagRequest request)
        {
            if (request is null)
                throw new NullTagViewRequestException();

            if (String.IsNullOrWhiteSpace(request.OldTag))
                throw new InvalidTagViewException("The tag being replaced cannot be null, empty, or whitespace.");

            if (String.IsNullOrWhiteSpace(request.NewTag))
                throw new InvalidTagViewException("The new tag cannot be null, empty, or whitespace.");
        }

        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullTagViewResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullTagViewResponseException.
            if (response == null)
            {
                throw new NullTagViewResponseException();
            }
        }
    }
}
