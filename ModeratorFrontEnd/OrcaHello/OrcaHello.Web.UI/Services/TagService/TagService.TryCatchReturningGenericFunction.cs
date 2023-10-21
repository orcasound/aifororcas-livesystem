namespace OrcaHello.Web.UI.Services
{
    public partial class TagService
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
                if (exception is InvalidTagException)
                    throw LoggingUtilities.CreateAndLogException<TagValidationException>(_logger, exception);

                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<TagDependencyValidationException>(_logger, new AlreadyExistsTagException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<TagDependencyValidationException>(_logger, new InvalidTagException(exception));

                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseInternalServerErrorException ||
                    exception is HttpResponseException)
                    throw LoggingUtilities.CreateAndLogException<TagDependencyException>(_logger, new FailedTagDependencyException(exception));

                throw LoggingUtilities.CreateAndLogException<TagServiceException>(_logger, new FailedTagDependencyException(exception));
            }
        }
    }
}
