import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.6, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.rect = None
        self.min_val = None
        self.max_val = None
        self.min_loc = None
        self.max_loc = None

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        #to get the first image
        if self.reference is None:
            self.reference = self.detect_face(image)
            if self.reference is None:
                print ("none")
        else:
            #print(self.reference)

            self.rect = self.reference["rect"].copy()
            #taking template from reference
            ROI = self.crop_face(image, self.rect)
            template = self.crop_face(self.reference["image"], self.reference["rect"])
            #template match
            res = cv2.matchTemplate(ROI, template, cv2.TM_CCOEFF_NORMED)
            self.min_val, self.max_val, self.min_loc, self.max_loc = cv2.minMaxLoc(res)
            print(self.max_val)
            if self.max_val < self.tm_threshold:
                # reinitialize MTCNN + reference = template
                self.reference = self.detect_face(image)
                if self.reference is None:
                    return None
            else:
                x, y = self.max_loc
                #to move the window to Image coordinates
                x += self.rect[0]
                y += self.rect[1]
                top_left = (x, y)
                bottom_right = (self.rect[2], self.rect[3])
                face_rect = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                aligned = self.align_face(image, face_rect)
                self.reference = {"rect": face_rect, "image": image, "aligned": aligned, "response": self.max_val}

        return self.reference



    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

