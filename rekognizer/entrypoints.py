import json
import types

from marshmallow import ValidationError
from nameko.exceptions import safe_for_serialization, BadRequest
from werkzeug import Response

from rekognizer.exceptions import (
    NoFaceException,
    TooManyFacesException,
    UnknownPersonException,
)

from functools import partial
from nameko.web.handlers import HttpRequestHandler
from nameko.extensions import register_entrypoint


class CorsHttpRequestHandler(HttpRequestHandler):
    """ Overrides `response_from_exception` so we can customize error handling.
    """

    mapped_errors = {
        BadRequest: (400, "BAD_REQUEST"),
        ValidationError: (400, "VALIDATION_ERROR"),
        NoFaceException: (400, "NO_FACE"),
        TooManyFacesException: (400, "TOO_MANY_FACE"),
        UnknownPersonException: (400, "UNKNOWN_PERSON"),
    }

    def response_from_exception(self, exc):
        status_code, error_code = 500, "UNEXPECTED_ERROR"

        if isinstance(exc, self.expected_exceptions):
            if type(exc) in self.mapped_errors:
                status_code, error_code = self.mapped_errors[type(exc)]
            else:
                status_code = 400
                error_code = "BAD_REQUEST"

        return Response(
            json.dumps({"error": error_code, "message": safe_for_serialization(exc)}),
            status=status_code,
            mimetype="application/json",
        )

    """
    A Cors Http handler. 
    Registers an OPTIONS route per endpoint definition.
    """
    def __init__(self, method, url, expected_exceptions=(), **kwargs):
        super().__init__(method, url, expected_exceptions=expected_exceptions)
        self.allowed_origin = kwargs.get('origin', ['*'])
        self.allowed_methods = kwargs.get('methods', ['*'])

    def handle_request(self, request):
        self.request = request
        if request.method == 'OPTIONS':
            return self.response_from_result(result='')
        return super().handle_request(request)

    def response_from_result(self, *args, **kwargs):
        response = super(CorsHttpRequestHandler, self).response_from_result(*args, **kwargs)
        response.headers.add("Access-Control-Allow-Headers",
                             self.request.headers.get("Access-Control-Request-Headers"))
        response.headers.add("Access-Control-Allow-Methods", ",".join(self.allowed_methods))
        response.headers.add("Access-Control-Allow-Origin", ",".join(self.allowed_origin))
        return response

    @classmethod
    def decorator(cls, *args, **kwargs):
        """
        We're overriding the decorator classmethod to allow it to register an options
        route for each standard REST call. This saves us from manually defining OPTIONS
        routes for each CORs enabled endpoint
        """
        def registering_decorator(fn, args, kwargs):
            instance = cls(*args, **kwargs)
            register_entrypoint(fn, instance)
            if instance.method in ('GET', 'POST', 'DELETE', 'PUT') and \
                    ('*' in instance.allowed_methods or instance.method in instance.allowed_methods):
                options_args = ['OPTIONS'] + list(args[1:])
                options_instance = cls(*options_args, **kwargs)
                register_entrypoint(fn, options_instance)
            return fn

        if len(args) == 1 and isinstance(args[0], types.FunctionType):
            # usage without arguments to the decorator:
            # @foobar
            # def spam():
            #     pass
            return registering_decorator(args[0], args=(), kwargs={})
        else:
            # usage with arguments to the decorator:
            # @foobar('shrub', ...)
            # def spam():
            #     pass
            return partial(registering_decorator, args=args, kwargs=kwargs)


http = CorsHttpRequestHandler.decorator
