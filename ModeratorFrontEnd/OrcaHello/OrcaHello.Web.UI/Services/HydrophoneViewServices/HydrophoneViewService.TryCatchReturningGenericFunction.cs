namespace OrcaHello.Web.UI.Services
{
    // This partial class implements a generic TryCatch for the HydrophoneViewService.
    public partial class HydrophoneViewService
    {
        // ReturningGenericFunction is a delegate that represents a generic asynchronous
        // function that returns a value of type T.
        public delegate ValueTask<T> ReturningGenericFunction<T>();

        // TryCatch is a method that takes a ReturningGenericFunction as a parameter and executes it in a try-catch block.
        // It handles different types of exceptions that may occur during the execution and logs them using LoggingUtilities.
        // It also rethrows the exceptions as specific types of HydrophoneViewServiceException,
        // HydrophoneViewDependencyException, HydrophoneViewDependencyValidationException, or
        // HydrophoneViewValidationException.
        protected async ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningGenericFunction)
        {
            try
            {
                // Try to execute the function and return the result.
                return await returningGenericFunction();
            }
            catch (Exception exception)
            {
                // If the exception is related to the validation of the hydrophone view, rethrow
                // it as a HydrophoneViewValidationException and log it.
                if (exception is NullHydrophoneViewException ||
                    exception is InvalidHydrophoneViewException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewValidationException>(_logger, exception);

                // If the exception is related to the validation of the hydrophone dependency, rethrow
                // it as a HydrophoneViewDependencyValidationException and log it.
                if (exception is HydrophoneValidationException ||
                    exception is HydrophoneDependencyValidationException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewDependencyValidationException>(_logger, exception);

                // If the exception is related to the dependency of the hydrophone service, rethrow
                // it as a HydrophoneViewDependencyException and log it.
                if (exception is HydrophoneDependencyException ||
                    exception is HydrophoneServiceException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneViewDependencyException>(_logger, exception);

                // If the exception is any other type, rethrow it as a HydrophoneViewServiceException and log it.
                throw LoggingUtilities.CreateAndLogException<HydrophoneViewServiceException>(_logger, new FailedHydrophoneViewServiceException(exception));
            }
        }
    }
}