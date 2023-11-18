﻿namespace OrcaHello.Web.UI
{
    public partial class StudentViewService
    {
        private delegate ValueTask<StudentView> ReturningStudentViewFunction();
        private delegate ValueTask<List<StudentView>> ReturningStudentViewsFunction();
        private delegate void ReturningNothingFunction();

        private async ValueTask<StudentView> TryCatch(ReturningStudentViewFunction returningStudentViewFunction)
        {
            try
            {
                return await returningStudentViewFunction();
            }
            catch (NullStudentViewException nullStudentViewException)
            {
                throw CreateAndLogValidationException(nullStudentViewException);
            }
            catch (InvalidStudentViewException invalidStudentViewException)
            {
                throw CreateAndLogValidationException(invalidStudentViewException);
            }
            catch (StudentValidationException studentValidationException)
            {
                throw CreateAndLogDependencyValidationException(studentValidationException);
            }
            catch (StudentDependencyValidationException studentDependencyValidationException)
            {
                throw CreateAndLogDependencyValidationException(studentDependencyValidationException);
            }
            catch (StudentDependencyException studentDependencyException)
            {
                throw CreateAndLogDependencyException(studentDependencyException);
            }
            catch (StudentServiceException studentServiceException)
            {
                throw CreateAndLogDependencyException(studentServiceException);
            }
            catch (Exception serviceException)
            {
                throw CreateAndLogServiceException(serviceException);
            }
        }

        private void TryCatch(ReturningNothingFunction returningNothingFunction)
        {
            try
            {
                returningNothingFunction();
            }
            catch (InvalidStudentViewException invalidStudentViewException)
            {
                throw CreateAndLogValidationException(invalidStudentViewException);
            }
            catch (Exception serviceException)
            {
                throw CreateAndLogServiceException(serviceException);
            }
        }

        private async ValueTask<List<StudentView>> TryCatch(ReturningStudentViewsFunction returningStudentViewsFunction)
        {
            try
            {
                return await returningStudentViewsFunction();
            }
            catch (StudentDependencyException studentDependencyException)
            {
                throw CreateAndLogDependencyException(studentDependencyException);
            }
            catch (StudentServiceException studentServiceException)
            {
                throw CreateAndLogDependencyException(studentServiceException);
            }
            catch (Exception serviceException)
            {
                var failedStudentViewServiceException = new FailedStudentViewServiceException(serviceException);

                throw CreateAndLogServiceException(failedStudentViewServiceException);
            }
        }

        private StudentViewValidationException CreateAndLogValidationException(Exception exception)
        {
            var studentViewValidationException = new StudentViewValidationException(exception);
            this.loggingBroker.LogError(studentViewValidationException);

            return studentViewValidationException;
        }

        private StudentViewDependencyValidationException CreateAndLogDependencyValidationException(Exception exception)
        {
            var studentViewDependencyValidationException =
                new StudentViewDependencyValidationException(exception.InnerException);

            this.loggingBroker.LogError(studentViewDependencyValidationException);

            return studentViewDependencyValidationException;
        }

        private StudentViewDependencyException CreateAndLogDependencyException(Exception exception)
        {
            var studentViewDependencyException = new StudentViewDependencyException(exception);
            this.loggingBroker.LogError(studentViewDependencyException);

            return studentViewDependencyException;
        }

        private StudentViewServiceException CreateAndLogServiceException(Exception exception)
        {
            var studentViewServiceException = new StudentViewServiceException(exception);
            this.loggingBroker.LogError(studentViewServiceException);

            return studentViewServiceException;
        }
    }
}
