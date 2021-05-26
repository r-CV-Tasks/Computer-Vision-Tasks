import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_faces(source: np.ndarray, scale_factor: float = 1.1, min_size: int = 50) -> list:
    """
    Detect faces in given image using xml cascades file which contains OpenCV data used to detect objects

    - loads the face cascade into memory

    :param source:
    :param min_size:
    :return: faces list which contains a lists of:
        - x, y location and width, height of each detected face
    """

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Create the haar cascade
    repo_root = os.path.dirname(os.getcwd())
    sys.path.append(repo_root)

    cascade_path = "./src/haarcascade_frontalface_default.xml"

    # cascade_path = "../src/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image=src,
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def draw_faces(source: np.ndarray, faces: list, thickness: int = 10) -> np.ndarray:
    """
    Draw rectangle around each face in the given faces list

    :param source:
    :param faces:
    :param thickness:
    :return:
    """

    src = np.copy(source)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h),
                      color=(0, 255, 0), thickness=thickness)

    return src


if __name__ == "__main__":
    image = cv2.imread("../resources/Images/faces/persons_image.jpg")
    # image = cv2.imread("../resources/Images/faces/IMG_8117.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_faces = detect_faces(source=image)
    faced_image = draw_faces(source=image_rgb, faces=detected_faces)

    print(f"Found {len(detected_faces)} Faces!")
    print(detected_faces)

    plt.imshow(faced_image)
    plt.show()
