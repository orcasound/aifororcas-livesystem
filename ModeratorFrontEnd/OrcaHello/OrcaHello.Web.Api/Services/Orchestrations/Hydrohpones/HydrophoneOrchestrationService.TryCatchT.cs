namespace OrcaHello.Web.Api.Services
{
    public partial class HydrophoneOrchestrationService
    {
        public delegate T ReturningGenericFunction<T>();

        protected T TryCatch<T>(ReturningGenericFunction<T> returningGenericFunction)
        {
            try
            {
                return returningGenericFunction();
            }
            catch (Exception exception)
            {
                if (exception is InvalidHydrophoneOrchestrationException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneOrchestrationValidationException>(_logger, exception);

                throw LoggingUtilities.CreateAndLogException<HydrophoneOrchestrationServiceException>(_logger, exception);

            }
        }
    }
}
