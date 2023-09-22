namespace OrcaHello.Web.UI.Services
{
    public partial class DetectionViewService
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
                if (exception is InvalidDetectionViewException)
                    throw LoggingUtilities.CreateAndLogException<DetectionViewValidationException>(_logger, exception);

                if (exception is DetectionValidationException ||
                    exception is DetectionDependencyValidationException)
                    throw LoggingUtilities.CreateAndLogException<DetectionViewDependencyValidationException>(_logger, exception);

                if (exception is DetectionDependencyException ||
                    exception is DetectionServiceException)
                    throw LoggingUtilities.CreateAndLogException<DetectionViewDependencyException>(_logger, exception);

                throw LoggingUtilities.CreateAndLogException<DetectionViewServiceException>(_logger, exception);
            }
        }
    }
}