from http import HTTPStatus


class AppException(Exception):
    status_code: int = HTTPStatus.BAD_REQUEST

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class NotFoundException(AppException):
    status_code: int = HTTPStatus.NOT_FOUND


class AlreadyExistsException(AppException):
    status_code: int = HTTPStatus.CONFLICT


class UnauthorizedException(AppException):
    status_code: int = HTTPStatus.UNAUTHORIZED


class ForbiddenException(AppException):
    status_code: int = HTTPStatus.FORBIDDEN


class BadRequestException(AppException):
    status_code: int = HTTPStatus.BAD_REQUEST
