from nameko.rpc import rpc
from nameko_sqlalchemy import DatabaseSession

from rekognizer.models import DeclarativeBase


class RekognizerService:
    name = "rekognizer"

    db = DatabaseSession(DeclarativeBase)

    @rpc
    def hello(self, name):
        return "Hello, {}!".format(name)
