import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_faces(source: np.ndarray) -> list:
    """
    Detect faces in given image using xlm cascades file which contains OpenCV data used to detect objects

    - loads the face cascade into memory

    :param source:
    :return: faces list which contains a lists of:
        - x, y location and width, height of each detected face
    """

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        pass

    # Create the haar cascade
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image=src,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def draw_faces(source: np.ndarray, faces: list) -> np.ndarray:
    """
    Draw rectangle around each face in the given faces list

    :param source:
    :param faces:
    :return:
    """

    src = np.copy(source)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return src


if __name__ == "__main__":
    image = cv2.imread("../resources/Images/persons_image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_faces = detect_faces(source=image)
    faced_image = draw_faces(source=image_rgb, faces=detected_faces)

    print(f"Found {len(detected_faces)} Faces!")
    print(detected_faces)

    plt.imshow(faced_image)
    plt.show()
