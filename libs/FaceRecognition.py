import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from libs import FaceDetection


def test_recognition():

    # know_face_encoding = None
    # unknown_face_encoding = None

    # know_image = face_recognition.load_image_file("../resources/Images/faces/IMG_8117.jpeg")
    know_image = face_recognition.load_image_file("../resources/Images/faces/IMG_20210407_170037.jpg")
    # know_image = face_recognition.load_image_file("../resources/Images/faces/refaey.jpeg")
    unknown_image = face_recognition.load_image_file("../resources/Images/faces/refaey2.jpg")

    know_face_encoding = face_recognition.face_encodings(face_image=know_image,
                                                         known_face_locations=detected_faces)

    print(f"Found {len(know_face_encoding)} faces in main image!")
    # print(know_face_encoding[0])
    print("---------------")

    unknown_face_encoding = face_recognition.face_encodings(face_image=unknown_image)
    print(f"Found {len(unknown_face_encoding)} faces in test image!")
    # print(unknown_face_encoding[0])
    print("---------------")



    # try:
    #     know_face_encoding = face_recognition.face_encodings(know_image)[0]
    #     unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    # except IndexError:
    #     print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    #     quit()
    #
    known_faces = [
        know_face_encoding,
    ]

    results = face_recognition.compare_faces(know_face_encoding, unknown_face_encoding[0])
    print(results)
    print(len(results))
    print("---------------")
    #
    # if (results[0]):
    #     print("I found Guido in the image")
    # else:
    #     print("Unknown person")
    pass


if __name__ == "__main__":
    # image = cv2.imread("../resources/Images/faces/persons_image.jpg")
    image = cv2.imread("../resources/Images/faces/IMG_8117.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detected_faces = FaceDetection.detect_faces(source=image)
    faced_image = FaceDetection.draw_faces(source=image_rgb, faces=detected_faces)

    print(f"Found {len(detected_faces)} Faces!")
    print(detected_faces)

    plt.imshow(faced_image)
    plt.show()
