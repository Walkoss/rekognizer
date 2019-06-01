from marshmallow import Schema, fields


class VerifySchema(Schema):
    image_urls = fields.List(fields.Url, required=True)
