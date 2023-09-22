namespace OrcaHello.Web.UI.Services
{
    public partial class DetectionService
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
                if (exception is InvalidDetectionException)
                    throw LoggingUtilities.CreateAndLogException<DetectionValidationException>(_logger, exception);

                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<DetectionDependencyValidationException>(_logger, new AlreadyExistsDetectionException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<DetectionDependencyValidationException>(_logger, new InvalidDetectionException(exception));

                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseInternalServerErrorException ||
                    exception is HttpResponseException)
                    throw LoggingUtilities.CreateAndLogException<DetectionDependencyException>(_logger, new FailedDetectionDependencyException(exception));



                throw LoggingUtilities.CreateAndLogException<DetectionServiceException>(_logger, new FailedDetectionDependencyException(exception));
            }
        }
    }
}
