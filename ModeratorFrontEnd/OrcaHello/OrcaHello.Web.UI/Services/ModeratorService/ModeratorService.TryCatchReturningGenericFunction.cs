using System;

namespace OrcaHello.Web.UI.Services
{

    // This partial class implements a generic TryCatch for the ModeratorService.
    public partial class ModeratorService
    {
        // ReturningGenericFunction is a delegate that represents a generic asynchronous
        // function that returns a value of type T.
        public delegate ValueTask<T> ReturningGenericFunction<T>();

        // TryCatch is a method that takes a ReturningGenericFunction as a parameter and executes it in a try-catch block.
        // It handles different types of exceptions that may occur during the execution and logs them using LoggingUtilities.
        // It also rethrows the exceptions as specific types of ModeratorServiceException, ModeratorDependencyException,
        // ModeratorDependencyValidationException, or ModeratorValidationException.
        protected async ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningGenericFunction)
        {
            try
            {
                // Try to execute the function and return the result.
                return await returningGenericFunction();
            }
            catch (Exception exception)
            {
                // If the exception is related to the validation of the response, rethrow
                // it as a ModeratorValidationException and log it.
                if (exception is NullModeratorResponseException ||
                    exception is InvalidModeratorException)
                    throw LoggingUtilities.CreateAndLogException<ModeratorValidationException>(_logger, exception);

                // If the exception is related to the dependency of the broker, rethrow
                // it as a ModeratorDependencyException and log it.
                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseException ||
                    exception is HttpResponseInternalServerErrorException)
                    throw LoggingUtilities.CreateAndLogException<ModeratorDependencyException>(_logger, new FailedModeratorDependencyException(exception));

                // If the exception is related to the conflict or bad request of the broker, rethrow
                // it as a ModeratorDependencyValidationException and log it.
                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<ModeratorDependencyValidationException>(_logger, new AlreadyExistsModeratorException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<ModeratorDependencyValidationException>(_logger, new InvalidModeratorException(exception));

                // If the exception is any other type, rethrow it as a HydrophoneServiceException and log it.
                throw LoggingUtilities.CreateAndLogException<ModeratorServiceException>(_logger, new FailedModeratorServiceException(exception));
            }
        }
    }
}