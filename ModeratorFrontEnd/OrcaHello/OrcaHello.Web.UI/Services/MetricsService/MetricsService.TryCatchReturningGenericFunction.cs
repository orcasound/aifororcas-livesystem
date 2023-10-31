namespace OrcaHello.Web.UI.Services
{
    public partial class MetricsService
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
                if (exception is InvalidMetricsException)
                    throw LoggingUtilities.CreateAndLogException<MetricsValidationException>(_logger, exception);

                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<MetricsDependencyValidationException>(_logger, new AlreadyExistsMetricsException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<MetricsDependencyValidationException>(_logger, new InvalidMetricsException(exception));

                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseInternalServerErrorException ||
                    exception is HttpResponseException)
                    throw LoggingUtilities.CreateAndLogException<MetricsDependencyException>(_logger, new FailedMetricsDependencyException(exception));



                throw LoggingUtilities.CreateAndLogException<MetricsServiceException>(_logger, new FailedMetricsDependencyException(exception));
            }

        }
    }
}
