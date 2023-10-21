namespace OrcaHello.Web.UI.Services
{
    public partial class MetricsViewService
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
                if (exception is InvalidMetricsViewException)
                    throw LoggingUtilities.CreateAndLogException<MetricsViewValidationException>(_logger, exception);

                if (exception is DetectionValidationException ||
                    exception is DetectionDependencyValidationException)
                    throw LoggingUtilities.CreateAndLogException<MetricsViewDependencyValidationException>(_logger, exception);

                if (exception is DetectionDependencyException ||
                    exception is DetectionServiceException)
                    throw LoggingUtilities.CreateAndLogException<MetricsViewDependencyException>(_logger, exception);

                throw LoggingUtilities.CreateAndLogException<MetricsViewServiceException>(_logger, exception);
            }
        }
    }
}