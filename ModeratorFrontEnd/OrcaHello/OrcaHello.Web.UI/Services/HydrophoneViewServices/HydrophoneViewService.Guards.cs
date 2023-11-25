namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="HydrophoneViewService"/> orchestration service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class HydrophoneViewService
    {
        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullHydrophoneViewResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullHydrophoneViewResponseException.
            if (response == null)
            {
                throw new NullHydrophoneViewResponseException();
            }
        }
    }
}
