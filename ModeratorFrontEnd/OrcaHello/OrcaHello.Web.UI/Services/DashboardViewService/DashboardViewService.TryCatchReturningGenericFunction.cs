namespace OrcaHello.Web.UI.Services
{
    public partial class DashboardViewService
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
                if (exception is InvalidDashboardViewException ||
                    exception is NullDashboardViewRequestException ||
                    exception is NullDashboardViewResponseException)
                    throw LoggingUtilities.CreateAndLogException<DashboardViewValidationException>(_logger, exception);

                if (exception is DetectionValidationException ||
                    exception is DetectionDependencyValidationException ||
                    exception is TagValidationException ||
                    exception is TagDependencyValidationException ||
                    exception is MetricsValidationException ||
                    exception is MetricsDependencyValidationException ||
                    exception is CommentValidationException ||
                    exception is CommentDependencyValidationException ||
                    exception is ModeratorValidationException ||
                    exception is ModeratorDependencyValidationException)
                    throw LoggingUtilities.CreateAndLogException<DashboardViewDependencyValidationException>(_logger, exception);

                if (exception is DetectionDependencyException ||
                    exception is DetectionServiceException ||
                    exception is TagDependencyException ||
                    exception is TagServiceException ||
                    exception is MetricsDependencyException ||
                    exception is MetricsServiceException ||
                    exception is CommentDependencyException ||
                    exception is CommentServiceException ||
                    exception is ModeratorDependencyException ||
                    exception is ModeratorServiceException)
                    throw LoggingUtilities.CreateAndLogException<DashboardViewDependencyException>(_logger, exception);

                throw LoggingUtilities.CreateAndLogException<DashboardViewServiceException>(_logger, exception);
            }
        }
    }
}