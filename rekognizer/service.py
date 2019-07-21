import json
import logging
from typing import List

import cv2
import numpy as np
from marshmallow import ValidationError
from nameko.rpc import rpc, RpcProxy
from nameko.events import EventDispatcher
from nameko.exceptions import BadRequest
from nameko_sqlalchemy import Database
from werkzeug.wrappers import Response

from rekognizer.entrypoints import http
from rekognizer.exceptions import (
    NoFaceException,
    TooManyFacesException,
    UnknownPersonException,
    UserDisabledException,
)
from rekognizer.face_detector import FaceDetector
from rekognizer.facenet import Facenet
from rekognizer.models import DeclarativeBase, Enrollment
from rekognizer.schema import VerifySchema, IdentifySchema
from rekognizer.utils import read_image, normalize_image, resize_image


class RekognizerService:
    name = "rekognizer"

    db = Database(DeclarativeBase)

    @rpc
    def enroll_user(self, user_id, image_urls):
        for image_url in image_urls:
            logging.info(f"Analyzing image {image_url}")
            image = read_image(image_url)
            if image.shape[1] > image.shape[0]:
                image = resize_image(image, width=600)
            else:
                image = resize_image(image, height=600)
            faces = FaceDetector.detect_faces(image)
            faces_len = len(faces)

            if faces_len == 0:
                raise NoFaceException(f"Image doesn't contains face")
            elif faces_len > 1:
                raise TooManyFacesException(f"Image contains too many faces")

            x, y, width, height = faces[0]["box"]
            cropped_image_face = image[y : y + height, x : x + width]
            cropped_image_face = cv2.resize(cropped_image_face, (160, 160))
            cropped_image_face = normalize_image(cropped_image_face)

            # Get embedding of image's face
            logging.info(f"Getting embedding {image_url}")
            embedding = Facenet.get_embeddings(np.array([cropped_image_face]))[0]

            with self.db.get_session() as session:
                session.add(Enrollment(embedding=embedding, user_id=user_id))


class RekognizerHttpService:
    name = "rekognizer_http"

    db = Database(DeclarativeBase)
    dispatch = EventDispatcher()
    user_manager = RpcProxy("user_manager")

    @http("POST", "/verify", expected_exceptions=(ValidationError, BadRequest))
    def verify(self, request):
        schema = VerifySchema(strict=True)

        try:
            verify_data = schema.loads(request.get_data(as_text=True)).data
        except ValueError as exc:
            raise BadRequest("Invalid json: {}".format(exc))

        verify_result = self._verify(verify_data["image_urls"])

        return Response(json.dumps(verify_result), mimetype="application/json")

    @http(
        "POST",
        "/identify",
        expected_exceptions=(
            ValidationError,
            BadRequest,
            NoFaceException,
            TooManyFacesException,
            UnknownPersonException,
            UserDisabledException,
        ),
    )
    def identify(self, request):
        schema = IdentifySchema(strict=True)

        try:
            identify_data = schema.loads(request.get_data(as_text=True)).data
        except ValueError as exc:
            raise BadRequest("Invalid json: {}".format(exc))

        identify_result = self._identify(identify_data["image_url"])

        return Response(json.dumps(identify_result), mimetype="application/json")

    def _verify(self, image_urls: List[str]):
        logging.info(f"Verifying urls: {image_urls}")

        result = []
        valid_images = []

        for image_url in image_urls:
            image = read_image(image_url)
            if image.shape[1] > image.shape[0]:
                image = resize_image(image, width=600)
            else:
                image = resize_image(image, height=600)
            faces = FaceDetector.detect_faces(image)
            faces_len = len(faces)

            if faces_len == 0:
                result.append(
                    {"image_url": image_url, "valid": False, "error": "NO_FACE"}
                )
                continue
            elif faces_len > 1:
                result.append(
                    {"image_url": image_url, "valid": False, "error": "TOO_MANY_FACES"}
                )
                continue

            x, y, width, height = faces[0]["box"]
            cropped_image_face = image[y : y + height, x : x + width]
            cropped_image_face = cv2.resize(cropped_image_face, (160, 160))
            cropped_image_face = normalize_image(cropped_image_face)

            valid_images.append({"image_url": image_url, "face": cropped_image_face})

        if len(valid_images) > 0:
            embeddings = Facenet.get_embeddings(
                np.array([image["face"] for image in valid_images])
            )
            similarities = Facenet.get_similarities(embeddings)
            for valid_image, similarity_result in zip(valid_images, similarities):
                if not similarity_result:
                    result.append(
                        {
                            "image_url": valid_image["image_url"],
                            "valid": False,
                            "error": "NOT_SAME_PERSON",
                        }
                    )
                else:
                    result.append(
                        {"image_url": valid_image["image_url"], "valid": True}
                    )

        return result

    def _identify(self, image_url: str):
        logging.info(f"Identifying url: {image_url}")

        image = read_image(image_url)
        if image.shape[1] > image.shape[0]:
            image = resize_image(image, width=600)
        else:
            image = resize_image(image, height=600)
        faces = FaceDetector.detect_faces(image)
        faces_len = len(faces)

        if faces_len == 0:
            raise NoFaceException(f"Image doesn't contains face")
        elif faces_len > 1:
            raise TooManyFacesException(f"Image contains too many faces")

        x, y, width, height = faces[0]["box"]
        cropped_image_face = image[y : y + height, x : x + width]
        cropped_image_face = cv2.resize(cropped_image_face, (160, 160))
        cropped_image_face = normalize_image(cropped_image_face)

        # Get embedding of image's face
        embedding = Facenet.get_embeddings(np.array([cropped_image_face]))[0]

        # Get all embeddings from database
        enrollments = self.db.session.query(Enrollment).all()

        if len(enrollments) == 0:
            raise UnknownPersonException(f"Image has not been identified")

        embeddings = np.array([eb.embedding for eb in enrollments])
        # Insert embedding at the beginning to compare with other embeddings
        embeddings = np.insert(embeddings, 0, embedding, axis=0)

        similarities = Facenet.get_similarities(embeddings)
        # Remove first element as it will always be True
        similarities.pop(0)

        try:
            index = similarities.index(True)

            user = self.user_manager.get_user(user_id=enrollments[index].user_id)
            if user["is_activated"] is False:
                raise UserDisabledException(f"User {user['id']} is disabled")

            self.dispatch("identification", user)

            return user
        except ValueError:
            raise UnknownPersonException(f"Image has not been identified")
