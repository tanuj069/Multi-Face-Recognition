"""
Create face embeddings for all the faces in the dataset/train directory
"""

import pickle
from imutils import paths
import cv2
import face_recognition
import os
from parameters import DLIB_FACE_ENCODING_PATH, DATASET_PATH


def create_face_embeddings():
    """
    This function creates face encodings for all the faces in the dataset/train directory
    """
    image_paths = list(paths.list_images(DATASET_PATH))
    print(image_paths)

    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []

    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        name = imagePath.split(os.path.sep)[-2]
        print(name)
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)

        encoding = face_recognition.face_encodings(
            image,
            num_jitters=10,  # Higher number of jitters increases the accuracy of the encoding
            model='large'  # model='large' or 'small'
        )[0]
        known_encodings.append(encoding)
        known_names.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    f = open(DLIB_FACE_ENCODING_PATH, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    create_face_embeddings()
