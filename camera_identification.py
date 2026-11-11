"""Identify an enrolled person with the existing camera model and distance gate."""

from skimage import data
from skimage.feature import Cascade

from camera_face import camera_features, locate_face, open_face_image
from verification import transform_features, verify_vector


def identify_camera_face(source, camera):
    """Return the accepted identity and an explanation for a rejected capture."""
    try:
        image = open_face_image(source)
        box = locate_face(image, Cascade(data.lbp_frontal_face_cascade_filename()))
        if box is None:
            return None, "No face detected. Clear the photo and center your face."
        vector = transform_features(camera["preprocessors"]["face"], camera_features(image, box))
        algorithm = camera["selected"]["face"]
        classifier = camera["models"]["face"][algorithm]["classifier"]
        candidate = str(classifier.predict(vector.reshape(1, -1))[0])
        decision = verify_vector("face", vector, candidate, camera, algorithm)
        if decision["accepted"]:
            return candidate, None
    except OSError:
        return None, "Could not read the photo. Clear it and take another."
    except ValueError:
        return None, "Could not analyze the photo. Clear it and take another."
    return None, "Face not recognized. Clear the photo and try again."
