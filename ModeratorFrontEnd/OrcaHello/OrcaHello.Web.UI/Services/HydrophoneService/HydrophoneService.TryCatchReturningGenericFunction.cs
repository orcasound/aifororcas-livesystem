namespace OrcaHello.Web.UI.Services
{
    // This partial class implements a generic TryCatch for the HydrophoneService.
    public partial class HydrophoneService
    {
        // ReturningGenericFunction is a delegate that represents a generic asynchronous
        // function that returns a value of type T.
        public delegate ValueTask<T> ReturningGenericFunction<T>();

        // TryCatch is a method that takes a ReturningGenericFunction as a parameter and executes it in a try-catch block.
        // It handles different types of exceptions that may occur during the execution and logs them using LoggingUtilities.
        // It also rethrows the exceptions as specific types of HydrophoneServiceException, HydrophoneDependencyException,
        // HydrophoneDependencyValidationException, or HydrophoneValidationException.
        protected async ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningGenericFunction)
        {
            try
            {
                // Try to execute the function and return the result.
                return await returningGenericFunction();
            }
            catch (Exception exception)
            {
                // If the exception is related to the validation of the hydrophone response, rethrow
                // it as a HydrophoneValidationException and log it.
                if (exception is NullHydrophoneResponseException ||
                    exception is InvalidHydrophoneException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneValidationException>(_logger, exception);

                // If the exception is related to the dependency of the hydrophone broker, rethrow
                // it as a HydrophoneDependencyException and log it.
                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseException ||
                    exception is HttpResponseInternalServerErrorException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyException>(_logger, new FailedHydrophoneDependencyException(exception));

                // If the exception is related to the conflict or bad request of the hydrophone broker, rethrow
                // it as a HydrophoneDependencyValidationException and log it.
                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyValidationException>(_logger, new AlreadyExistsHydrophoneException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyValidationException>(_logger, new InvalidHydrophoneException(exception));

                // If the exception is any other type, rethrow it as a HydrophoneServiceException and log it.
                throw LoggingUtilities.CreateAndLogException<HydrophoneServiceException>(_logger, new FailedHydrophoneServiceException(exception));
            }
        }
    }
}