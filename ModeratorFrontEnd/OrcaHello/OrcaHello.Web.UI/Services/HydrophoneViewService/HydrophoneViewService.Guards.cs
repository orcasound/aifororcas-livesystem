namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="HydrophoneViewService"/> orchestration service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class HydrophoneViewService
    {
        // RULE: Response cannot be null.
        protected void ValidateResponse<T>(T response)
        {
            if (response == null)
            {
                throw new NullHydrophoneViewResponseException(nameof(T));
            }
        }
    }
}
