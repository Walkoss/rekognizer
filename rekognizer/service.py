import json
import logging
from typing import List

import cv2
import numpy as np
from marshmallow import ValidationError
from nameko.exceptions import BadRequest
from nameko.rpc import rpc, RpcProxy
from werkzeug.wrappers import Response

from rekognizer.entrypoints import http
from rekognizer.face_detector import FaceDetector
from rekognizer.facenet import Facenet
from rekognizer.schema import VerifySchema
from rekognizer.utils import read_image, normalize_image


class RekognizerHttpService:
    name = "rekognizer_http"

    rekognizer_rpc = RpcProxy("rekognizer")

    @http("POST", "/verify", expected_exceptions=(ValidationError, BadRequest))
    def verify(self, request):
        logging.info(request)
        schema = VerifySchema(strict=True)

        try:
            verify_data = schema.loads(request.get_data(as_text=True)).data
        except ValueError as exc:
            raise BadRequest("Invalid json: {}".format(exc))

        verify_result = self.rekognizer_rpc.verify(verify_data["image_urls"])

        return Response(json.dumps(verify_result), mimetype="application/json")


class RekognizerService:
    name = "rekognizer"

    @rpc
    def verify(self, image_urls: List[str]):
        logging.info(f"Verifying urls: {image_urls}")

        result = []
        valid_images = []

        for image_url in image_urls:
            image = read_image(image_url)
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
