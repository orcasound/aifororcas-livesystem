namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="HydrophoneService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class HydrophoneService
    {
        // RULE: Check if the response from the broker contains any hydrophones.
        protected void ValidateThereAreHydrophones(int count)
        {
            if (count == 0)
            {
                throw new InvalidHydrophoneException("Call returned no hydrophones.");
            }
        }

        // RULE: Response cannot be null.
        protected void ValidateResponse<T>(T response)
        {
            if (response == null)
            {
                throw new NullHydrophoneResponseException();
            }
        }
    }
}
