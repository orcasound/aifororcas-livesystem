namespace OrcaHello.Web.UI.Services
{ 
    public partial class CommentService
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
                if (exception is InvalidCommentException)
                    throw LoggingUtilities.CreateAndLogException<CommentValidationException>(_logger, exception);

                if (exception is HttpResponseConflictException)
                    throw LoggingUtilities.CreateAndLogException<CommentDependencyValidationException>(_logger, new AlreadyExistsCommentException(exception));

                if (exception is HttpResponseBadRequestException)
                    throw LoggingUtilities.CreateAndLogException<CommentDependencyValidationException>(_logger, new InvalidCommentException(exception));

                if (exception is HttpRequestException ||
                    exception is HttpResponseUrlNotFoundException ||
                    exception is HttpResponseUnauthorizedException ||
                    exception is HttpResponseInternalServerErrorException ||
                    exception is HttpResponseException)
                    throw LoggingUtilities.CreateAndLogException<CommentDependencyException>(_logger, new FailedCommentDependencyException(exception));



                throw LoggingUtilities.CreateAndLogException<CommentServiceException>(_logger, new FailedCommentDependencyException(exception));
            }

        }
    }
}
