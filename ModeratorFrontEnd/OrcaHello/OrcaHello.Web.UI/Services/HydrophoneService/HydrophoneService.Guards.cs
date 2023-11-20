namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="HydrophoneService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class HydrophoneService
    {
        // RULE: Check if the response from the broker is not null.
        // If the response is null, throw a NullHydrophoneResponseException.
        private static void ValidateResponse(HydrophoneListResponse response)
        {
            if(response == null)
            {
                throw new NullHydrophoneResponseException();
            }
        }

        // RULE: Check if the response from the broker contains any hydrophones.
        // If the response has zero hydrophones, throw an InvalidHydrophoneException with a custom message.
        private static void ValidateThereAreHydrophones(int count)
        {
            if (count == 0)
            {
                throw new InvalidHydrophoneException("Call returned no hydrophones.");
            }
        }
    }
}
