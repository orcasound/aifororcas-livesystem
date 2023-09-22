namespace OrcaHello.Web.UI.Services
{
    public partial class HydrophoneService
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
                if (exception is InvalidHydrophoneException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneValidationException>(_logger, exception);

                 if(exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyValidationException>(_logger, new AlreadyExistsHydrophoneException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyValidationException>(_logger, new InvalidHydrophoneException(exception));

                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseInternalServerErrorException ||
                    exception is HttpResponseException)
                    throw LoggingUtilities.CreateAndLogException<HydrophoneDependencyException>(_logger, new FailedHydrophoneDependencyException(exception));

                throw LoggingUtilities.CreateAndLogException<HydrophoneServiceException>(_logger, new FailedHydrophoneDependencyException(exception));
            }
        }
    }
}