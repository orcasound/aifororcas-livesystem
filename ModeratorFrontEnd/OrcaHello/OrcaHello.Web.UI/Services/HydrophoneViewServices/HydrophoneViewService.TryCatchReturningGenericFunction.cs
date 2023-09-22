namespace OrcaHello.Web.UI.Services
{
    public partial class HydrophoneViewService
    {
        public delegate ValueTask<T> ReturningGenericFunction<T>();

        protected async ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningGenericFunction)
        {
            try
            {
                return await returningGenericFunction();
            }
            catch (Exception exception)
            {
                if(exception is InvalidHydrophoneViewException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewValidationException>(_logger, exception);

                if(exception is HydrophoneValidationException ||
                    exception is HydrophoneDependencyValidationException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewDependencyValidationException>(_logger, exception);

                if(exception is HydrophoneDependencyException ||
                    exception is HydrophoneServiceException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewDependencyException>(_logger, exception);

                throw LoggingUtilities.CreateAndLogException<HydrophoneViewServiceException>(_logger, exception);
            }
        }
    }
}