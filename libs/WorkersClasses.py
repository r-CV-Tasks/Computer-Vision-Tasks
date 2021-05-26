import timeit
from typing import Callable, Type

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from libs import SIFT, FeatureMatching
from libs.FaceRecognition import FaceRecognizer


# Step 1: Create a worker class (subclass of QObject)
class SIFTWorker(QObject):
    def __init__(self, source: np.ndarray, source_id: int, start_time: float):
        """

        :param source:
        :param source_id:
        :param start_time:
        """
        super().__init__()
        self.img = source
        self.source_id = source_id
        self.start_time = start_time
        self.end_time = 0

    # Create 2 signals
    finished = pyqtSignal(list, np.ndarray, int, float)
    progress = pyqtSignal(int)

    def run(self):
        """
        Function to run a long task
        This is executed when calling SIFTWorker.start() in the main application

        :return:
        """

        # Apply SIFT Detector and Descriptor
        keypoints, descriptors = SIFT.Sift(src=self.img)

        # Function end
        end_time = timeit.default_timer()

        # # Show only 5 digits after floating point
        elapsed_time = float(format(end_time - self.start_time, '.5f'))

        # Emit finished signal to end the thread
        self.finished.emit(keypoints, descriptors, self.source_id, elapsed_time)


class MatchingWorker(QObject):
    def __init__(self, source1: np.ndarray, source2: np.ndarray,
                 desc1: np.ndarray, desc2: np.ndarray,
                 keypoints1: list, keypoints2: list,
                 match_calculator: Callable, num_matches: int,
                 source_id: int, start_time: float):
        """

        :param source1:
        :param source2:
        :param desc1:
        :param desc2:
        :param keypoints1:
        :param keypoints2:
        :param match_calculator:
        :param num_matches:
        :param source_id:
        :param start_time:
        """
        super().__init__()
        self.img1 = source1
        self.img2 = source2
        self.desc1 = desc1
        self.desc2 = desc2
        self.match_calculator = match_calculator
        self.keypoints1 = keypoints1
        self.keypoints2 = keypoints2
        self.num_matches = num_matches
        self.source_id = source_id
        self.start_time = start_time
        self.end_time = 0

    # Create 2 signals
    finished = pyqtSignal(np.ndarray, int, float)
    progress = pyqtSignal(int)

    def run(self):
        """
        Function to run a long task
        This is executed when calling MatchingWorker.start() in the main application
        :return:
        """

        matches = FeatureMatching.apply_feature_matching(desc1=self.desc1,
                                                         desc2=self.desc2,
                                                         match_calculator=self.match_calculator)

        matches = sorted(matches, key=lambda x: x.distance, reverse=True)

        matched_image = cv2.drawMatches(self.img1, self.keypoints1,
                                        self.img2, self.keypoints2,
                                        matches[:self.num_matches], self.img2, flags=2)

        # Function end
        end_time = timeit.default_timer()

        # # Show only 5 digits after floating point
        elapsed_time = float(format(end_time - self.start_time, '.5f'))

        # Emit finished signal to end the thread
        self.finished.emit(matched_image, self.source_id, elapsed_time)


class FilterWorker(QObject):
    def __init__(self, source: np.ndarray, filter_function: Callable, shape: int, combo_id: str,  sigma: float = 0.0):
        """

        :param source:
        :param shape:
        """
        super().__init__()
        self.noisy_image = source
        self.shape = shape
        self.sigma = sigma
        self.combo_id = combo_id
        self.filter_function = filter_function

    # Create 2 signals
    finished = pyqtSignal(np.ndarray, str)
    progress = pyqtSignal(int)

    def run(self):
        """
        Function to run a long task
        This is executed when calling FilterWorker.start() in the main application
        :return:
        """
        filtered_image = self.filter_function(source=self.noisy_image, shape=self.shape, sigma=self.sigma)

        # Emit finished signal to end the thread
        self.finished.emit(filtered_image, self.combo_id)

class FaceRecognitionWorker(QObject):

    # Create 2 signals
    finished = pyqtSignal(str, float)
    progress = pyqtSignal(int)

    def __init__(self, recognizer_obj: Type[Callable], test_path: str, source_id: int, start_time: float):
        """

        :param source1:
        :param source2:
        :param desc1:
        :param desc2:
        :param keypoints1:
        :param keypoints2:
        :param match_calculator:
        :param num_matches:
        :param source_id:
        :param start_time:
        """
        super().__init__()
        self.recognizer_obj = recognizer_obj
        self.test_path = test_path
        self.source_id = source_id
        self.start_time = start_time
        self.end_time = 0


    def run(self):
        """
        Function to run a long task
        This is executed when calling MatchingWorker.start() in the main application
        :return:
        """

        recognized_name = self.recognizer_obj.recognize_face(source_path=self.test_path)

        # Function end
        end_time = timeit.default_timer()

        # # Show only 5 digits after floating point
        elapsed_time = float(format(end_time - self.start_time, '.5f'))

        # Emit finished signal to end the thread
        self.finished.emit(recognized_name, elapsed_time)