# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com
# Class Style Edits: Harshith Thota

# import necessary packages
import keras
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
from tqdm import tqdm

# disable TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GenderDetector:
    def __init__(self):
        # download pre-trained model file (one-time download)
        dwnld_link = "https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model"
        model_path = get_file(
            "gender_detection.model",
            dwnld_link,
            cache_subdir="pre-trained",
            cache_dir=os.getcwd()
        )
        # load model
        self.model = load_model(model_path)

    @staticmethod
    def get_cropped_face(image, focal_points):
        # get corner points of face rectangle       
        (startX, startY) = focal_points[0:2]
        (endX, endY) = focal_points[2:4]
        # crop the detected face region
        face_crop = np.copy(image[startY:endY, startX:endX])
        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        return face_crop

    def detect_gender(self, byte_content):
        # read input image
        npimg = np.fromstring(byte_content, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        #image = cv2.imread(image)
        if image is None:
            return False # Could not read input image
        # detect faces in the image
        face, confidence = cv.detect_face(image, threshold=0.95)
        classes = ['man','woman']
        # loop through detected faces
        genders = []
        for idx, focal_points in enumerate(face):
            face_crop = self.get_cropped_face(image, focal_points)
            # apply gender detection on face
            conf = self.model.predict(face_crop)[0]
            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
            genders.append(label)
        return genders

if __name__ == "__main__":
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="path to input image"
    )
    args = ap.parse_args()

    gd = GenderDetector()
    genders = gd.detect_gender(input('Enter the path to the test image:\n'))
    print(genders)
