namespace OrcaHello.Web.UI.Services
{
    // This partial class implements all level-specific validations for the HydrophoneService.
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
