# Importing Packages
# importing module
import logging
import os
import sys
import timeit
from typing import Callable, Type

import cv2
import numpy as np
from PIL import Image
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, QFile, QTextStream
from PyQt5.QtWidgets import QMessageBox

from UI import mainGUI as m
from UI import breeze_resources
from libs import EdgeDetection, Noise, LowPass, Histogram, FrequencyFilters, \
    Hough, Contour, Harris, FeatureMatching, SegmentationThresholding, SegmentationClustering, \
    FaceDetection, FaceRecognition

from WorkersClasses import SIFTWorker, MatchingWorker, FilterWorker, FaceRecognitionWorker

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(lineno)s - %(levelname)s - %(message)s',
                    filemode='w')

# Creating a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ImageProcessor(m.Ui_MainWindow):
    """
    Main Class of the program GUI
    """

    def __init__(self, starter_window):
        """
        Main loop of the UI
        :param starter_window: QMainWindow Object
        """
        super(ImageProcessor, self).setupUi(starter_window)

        # Setup Main_TabWidget Connections
        self.Main_TabWidget.setCurrentIndex(0)
        self.tab_index = self.Main_TabWidget.currentIndex()
        self.Main_TabWidget.currentChanged.connect(self.tab_changed)

        # Images Lists
        self.inputImages = [[self.img0_input], [self.img1_input],
                            [self.img2_1_input, self.img2_2_input],
                            [self.img3_input], [self.img4_input],
                            [self.img5_input],
                            [self.img6_1_input, self.img6_2_input],
                            [self.img7_1_input, self.img7_2_input],
                            [self.img8_1_input],
                            [None, self.img9_2_input]]

        self.outputImages = [[self.img0_noisy, self.img0_filtered, self.img0_edged],
                             self.img1_output, self.img2_output, self.img3_output,
                             self.img4_output, self.img5_output, self.img6_output,
                             [self.img7_1_output, self.img7_2_output],
                             self.img8_1_output, self.img9_2_output]

        self.processedImages = {"0_1": self.img0_noisy, "0_2": self.img0_filtered, "0_3": self.img0_edged,
                                "7_1": self.img7_1_output, "7_2": self.img7_2_output,
                                "8_1": self.img8_1_output,
                                "9_1": None, "9_2": self.img9_2_output}

        self.histogramImages = {"1_1": self.img1_input_histo, "1_2": self.img1_output, "1_3": self.img1_output_histo}

        # This contains all the widgets to setup them in one loop
        self.imageWidgets = [self.img0_input, self.img0_noisy, self.img0_filtered, self.img0_edged,
                             self.img1_input, self.img1_output,
                             self.img2_1_input, self.img2_2_input, self.img2_output,
                             self.img3_input, self.img3_output,
                             self.img4_input, self.img4_output,
                             self.img5_input, self.img5_output,
                             self.img6_1_input, self.img6_2_input, self.img6_output,
                             self.img7_1_input, self.img7_2_input, self.img7_1_output, self.img7_2_output,
                             self.img8_1_input, self.img8_1_output,
                             self.img9_2_input, self.img9_2_output]

        # Initial Variables
        self.currentNoiseImage = None
        self.edged_image = None
        self.filtered_image = None
        self.output_hist_image = None
        self.updated_image = None
        self.db_path = None

        # Threads and workers we will use in QThread
        # Used for SIFT Algorithm, Feature Matching and Median Filter
        self.threads = {}
        self.workers = {}

        # SIFT Results
        self.sift_results = {}

        # Dictionaries to store images data
        self.imagesData = {}
        self.imagesPaths = {}
        self.widths = {}
        self.heights = {}

        # Object of FaceRecognizer Class
        self.recognizer = None

        # Images Labels and Sizes
        self.imagesLabels = {"0_1": self.label_imgName_0, "1_1": self.label_imgName_1,
                             "2_1": self.label_imgName_2_1, "2_2": self.label_imgName_2_2,
                             "3_1": self.label_imgName_3, "4_1": self.label_imgName_4,
                             "5_1": self.label_imgName_5,
                             "6_1": self.label_imgName_6_1, "6_2": self.label_imgName_6_2,
                             "7_1": self.label_imgName_7_1, "7_2": self.label_imgName_7_2,
                             "8_1": self.label_imgName_8_1,
                             "9_1": self.label_imgName_9_1, "9_2": self.label_imgName_9_2}

        self.imagesSizes = {"0_1": self.label_imgSize_0, "1_1": self.label_imgSize_1,
                            "2_1": self.label_imgSize_2_1, "2_2": self.label_imgSize_2_2,
                            "3_1": self.label_imgSize_3, "4_1": self.label_imgSize_4,
                            "5_1": self.label_imgSize_5,
                            "6_1": self.label_imgSize_6_1, "6_2": self.label_imgSize_6_2,
                            "7_1": self.label_imgSize_7_1, "7_2": self.label_imgSize_7_2,
                            "8_1": self.label_imgSize_8_1,
                            "9_1": self.label_imgSize_9_1, "9_2": self.label_imgSize_9_2}

        # list contains the last pressed values
        self.sliderValuesClicked = {0: ..., 1: ..., 2: ..., 3: ...}
        self.sliders = [self.snr_slider_1, self.sigma_slider_1, self.mask_size_1, self.sigma_slider_2]

        # Sliders Connections
        for slider in self.sliders:
            slider.id = self.sliders.index(slider)
            slider.signal.connect(self.slider_changed)

        # Combo Boxes Dictionary
        self.comboBoxes = {"0_1": self.combo_noise, "0_2": self.combo_filter,
                           "0_3": self.combo_edges,
                           "1_1": self.combo_histogram,
                           "6_1": self.combo_matching_methods,
                           "7_1": self.combo_thresholding_methods, "7_2": self.combo_clustering_methods}

        # Setup Load Buttons Connections
        self.btn_load_0_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_1_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2_2.clicked.connect(lambda: self.load_file(self.tab_index, True))
        self.btn_load_3_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_4_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_5_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_6_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_6_2.clicked.connect(lambda: self.load_file(self.tab_index, True))
        self.btn_load_7_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_7_2.clicked.connect(lambda: self.load_file(self.tab_index, True))

        # Face Detection
        self.btn_load_8_1.clicked.connect(lambda: self.load_file(self.tab_index))

        # Face Recognition
        # TODO: Refactor this to other specific function
        self.btn_load_9_1.clicked.connect(self.create_database)
        self.btn_load_9_2.clicked.connect(lambda: self.load_file(self.tab_index, True))

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index, combo_id="0_1"))
        self.combo_filter.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index, combo_id="0_2"))
        self.combo_edges.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index, combo_id="0_3"))
        self.combo_histogram.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index, combo_id="1_1"))
        self.combo_thresholding_methods.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index,
                                                                                         combo_id="7_1"))
        self.combo_clustering_methods.activated.connect(lambda: self.combo_box_changed(tab_id=self.tab_index,
                                                                                       combo_id="7_2"))

        # Setup Hybrid Button
        self.btn_hybrid.clicked.connect(self.hybrid_image)

        # Setup Hough Button
        self.btn_hough.clicked.connect(self.hough_transform)

        # Setup Active Contour Buttons
        self.btn_apply_contour.clicked.connect(self.active_contour)
        self.btn_clear_anchors.clicked.connect(self.clear_anchors)
        self.btn_reset_contour.clicked.connect(self.reset_contour)

        # Setup Harris Operator Button
        self.btn_apply_harris.clicked.connect(self.harris_operator)

        # Setup SIFT Button
        self.btn_match_features.clicked.connect(self.sift)

        # Setup Face Detection Buttons
        self.btn_detect_faces.clicked.connect(self.detect_faces)

        # Set Face Recognition Buttons
        self.btn_make_eigenfaces.clicked.connect(self.make_eigenfaces)
        self.btn_match_faces.clicked.connect(self.recognize_faces)

        self.setup_images_view()

    def tab_changed(self):
        """
        Updates the current tab index

        :return: void
        """

        self.tab_index = self.Main_TabWidget.currentIndex()

    def setup_images_view(self):
        """
        This function is responsible for:
            - Adjusting the shape and scales of the widgets
            - Remove unnecessary options

        :return: void
        """

        for widget in self.imageWidgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def load_file(self, img_id: int, multi_widget: bool = False):
        """
        Load the File from User

        :param img_id: current tab index
        :param multi_widget: Flag to check if the tab has more than one image
        :return:
        """

        # Open File Browser
        logger.info("Browsing the files...")
        repo_path = "resources/Images"
        filename, file_format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                      "*;;" "*.jpg;;" "*.jpeg;;" "*.png;;")

        # If the file is loaded successfully
        if filename != "":
            # Take last part of the filename string
            img_name = filename.split('/')[-1]

            # Read the image
            img_bgr = cv2.imread(filename)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Loading one image only in the tab
            if multi_widget is False:
                # Convert int index to key to use in the dictionaries
                img_idx = str(img_id) + "_1"
            # When Loading 2nd Image in same Tab (Hybrid, SIFT, Template Matching)
            else:
                img_idx = str(img_id) + "_2"

            # Store the image
            self.imagesData[img_idx] = img_rgb
            self.imagesPaths[img_idx] = filename
            self.heights[img_idx], self.widths[img_idx], _ = img_rgb.shape

            # Reset Results
            self.clear_results(tab_id=img_id, combo_id=multi_widget)

            # Display Original Image
            self.display_image(source=self.imagesData[img_idx], widget=self.inputImages[img_id][multi_widget])

            # Enable the comboBoxes and settings
            self.enable_gui(tab_id=img_id)

            # Set Image Name and Sizes
            self.imagesLabels[img_idx].setText(img_name)
            self.imagesSizes[img_idx].setText(f"{self.widths[img_idx]}x{self.heights[img_idx]}")

            logger.info(f"Added Image #{img_id}: {img_name} successfully")

        # The file wasn't loaded correctly
        else:
            self.show_message(header="Warning!!", message="You didn't choose any image",
                              button=QMessageBox.Ok, icon=QMessageBox.Warning)

    def enable_gui(self, tab_id: int):
        """
        This function enables the required elements in the gui
        :param tab_id: if of the current tab
        :return:
        """

        # in Filters Tab
        if tab_id == 0:
            # Enable ComboBoxes
            self.combo_noise.setEnabled(True)
            self.combo_edges.setEnabled(True)

        # in Histograms Tab
        elif tab_id == 1:
            self.combo_histogram.setEnabled(True)

        # in Hybrid Tab
        elif tab_id == 2:
            self.btn_load_2_2.setEnabled(True)
            try:
                if isinstance(self.imagesData["2_1"], np.ndarray) and isinstance(self.imagesData["2_2"], np.ndarray):
                    self.btn_hybrid.setEnabled(True)
            except KeyError:
                print("Load another Image to apply hybrid")

        # in Hough Tab
        elif tab_id == 3:
            self.hough_settings_layout.setEnabled(True)
            self.btn_hough.setEnabled(True)

        # in Active Contour Tab
        elif tab_id == 4:
            self.contour_settings_layout.setEnabled(True)
            self.btn_clear_anchors.setEnabled(True)
            self.btn_apply_contour.setEnabled(True)
            self.btn_reset_contour.setEnabled(True)

        # in Harris Operator Tab
        elif tab_id == 5:
            self.harris_settings_layout.setEnabled(True)
            self.btn_apply_harris.setEnabled(True)

        # in SIFT Tab
        elif tab_id == 6:
            self.btn_load_6_2.setEnabled(True)
            self.combo_matching_methods.setEnabled(True)
            self.label_matches_number.setEnabled(True)
            self.text_sift_matches.setEnabled(True)
            self.btn_match_features.setEnabled(True)

        # in Segmentation  Tab
        elif tab_id == 7:
            if "7_1" in self.imagesData:
                self.combo_thresholding_methods.setEnabled(True)
                self.label_regions_x_local.setEnabled(True)
                self.label_regions_y_local.setEnabled(True)
                self.text_regions_x_numbers.setEnabled(True)
                self.text_regions_y_numbers.setEnabled(True)

            if "7_2" in self.imagesData:
                self.combo_clustering_methods.setEnabled(True)
                self.label_clusters_number.setEnabled(True)
                self.label_clustering_threshold.setEnabled(True)
                self.text_clusters_numbers.setEnabled(True)
                self.text_clustering_threshold.setEnabled(True)

        # Face Detection Tab
        elif tab_id == 8:
            self.btn_detect_faces.setEnabled(True)
            self.text_rect_thickness.setEnabled(True)
            self.text_scale_factor.setEnabled(True)

        # Face Recognition Tab
        elif tab_id == 9:
            self.btn_match_faces.setEnabled(True)

    def clear_results(self, tab_id: int, combo_id: int = 0):
        """
        Clears previous results when loading a new image

        :param tab_id: current tab index
        :param combo_id:
        :return: void
        """

        # Check current tab index
        if tab_id == 0:
            # Clear Filters Images Widgets
            self.img0_noisy.clear()
            self.img0_filtered.clear()
            self.img0_edged.clear()

            # Reset combo boxes choices
            self.combo_noise.setCurrentIndex(0)
            self.combo_filter.setCurrentIndex(0)
            self.combo_edges.setCurrentIndex(0)

            # Reset Sliders Values
            self.snr_slider_1.setValue(5)
            self.sigma_slider_1.setValue(2)
            self.mask_size_1.setValue(2)
            self.sigma_slider_2.setValue(2)

            # Clear recently noised image
            self.currentNoiseImage = None

        elif tab_id == 1:
            # Clear Histograms Widgets
            for widget in self.histogramImages.values():
                widget.clear()

            self.clear_bar_graph(widget=self.histogramImages["1_3"])

            # Reset combo box choices
            self.combo_histogram.setCurrentIndex(0)

        elif tab_id == 2:
            self.img2_output.clear()

        elif tab_id == 3:
            self.img3_output.clear()

        elif tab_id == 4:
            self.img4_output.clear()

        elif tab_id == 5:
            self.img5_output.clear()

        elif tab_id == 6:
            self.img6_output.clear()

        elif tab_id == 7:
            # Don't clear Output of the other image
            if combo_id == 0:
                self.img7_1_output.clear()
                self.combo_thresholding_methods.setCurrentIndex(0)

            elif combo_id == 1:
                self.img7_2_output.clear()
                self.combo_clustering_methods.setCurrentIndex(0)

        # Face Detection Tab
        elif tab_id == 8:
            self.img8_1_output.clear()

        # Face Recognition Tab
        elif tab_id == 9:
            if combo_id == 1:
                self.label_elapsed_time_face_testing.setText("0.0")
                self.img9_2_output.clear()

    def combo_box_changed(self, tab_id: int, combo_id: str):
        """

        :param tab_id: id of the current tab
        :param combo_id: id of the chosen combo box
        :return:
        """

        # If tab 0: key will be "0_1" which exists in the dictionary
        img_key = str(tab_id) + "_1"

        # If 1st tab is selected
        if tab_id == 0:
            # Get Values from combo box and sliders
            selected_component = self.comboBoxes[combo_id].currentText().lower()

            # Adjust Sliders Values
            noise_snr = self.snr_slider_1.value() / 10
            noise_sigma = self.sigma_slider_1.value()  # This value from 0 -> 4 so we need to map to another range
            noise_sigma = np.round(np.interp(noise_sigma, [0, 4], [0, 255]))  # This value from 0 -> 255

            filter_sigma = self.sigma_slider_2.value()
            filter_sigma = np.round(np.interp(filter_sigma, [0, 4], [0, 255]))

            mask_size = self.mask_size_1.value()
            mask_size = int(np.round(np.interp(mask_size, [1, 4], [3, 9])))

            # Noise Options
            if combo_id == "0_1":
                # Enable SNR Slider and Filters Combo
                self.snr_slider_1.setEnabled(True)
                self.combo_filter.setEnabled(True)

                if selected_component == "uniform noise":
                    self.currentNoiseImage = Noise.uniform_noise(source=self.imagesData[img_key], snr=noise_snr)

                elif selected_component == "gaussian noise":
                    self.sigma_slider_1.setEnabled(True)
                    self.currentNoiseImage = Noise.gaussian_noise(source=self.imagesData[img_key], sigma=noise_sigma,
                                                                  snr=noise_snr)

                elif selected_component == "salt & pepper noise":
                    self.currentNoiseImage = Noise.salt_pepper_noise(source=self.imagesData[img_key], snr=noise_snr)

                try:
                    self.display_image(source=self.currentNoiseImage, widget=self.processedImages[combo_id])
                except TypeError:
                    print("Cannot display Noisy Image")

            # Filters Options
            if combo_id == "0_2":
                self.mask_size_1.setEnabled(True)

                # Check if there's a noisy image already
                if self.currentNoiseImage is None:
                    self.show_message(header="Warning!!", message="Apply noise to the image first",
                                      button=QMessageBox.Ok, icon=QMessageBox.Warning)

                elif selected_component == "average filter":
                    self.create_filter_thread(source=self.currentNoiseImage, source_id=2,
                                              filter_function=LowPass.average_filter, shape=mask_size,
                                              combo_id=combo_id)

                elif selected_component == "gaussian filter":
                    self.sigma_slider_2.setEnabled(True)

                    self.create_filter_thread(source=self.currentNoiseImage, source_id=2,
                                              filter_function=LowPass.gaussian_filter, shape=mask_size,
                                              combo_id=combo_id, sigma=filter_sigma)

                elif selected_component == "median filter":
                    self.create_filter_thread(source=self.currentNoiseImage, source_id=2,
                                              filter_function=LowPass.median_filter, shape=mask_size, combo_id=combo_id)

            # Edge Detection Options
            if combo_id == "0_3":
                if selected_component == "sobel mask":
                    self.edged_image = EdgeDetection.sobel_edge(source=self.imagesData[img_key])

                elif selected_component == "roberts mask":
                    self.edged_image = EdgeDetection.roberts_edge(source=self.imagesData[img_key])

                elif selected_component == "prewitt mask":
                    self.edged_image = EdgeDetection.prewitt_edge(source=self.imagesData[img_key])

                elif selected_component == "canny mask":
                    self.edged_image = EdgeDetection.canny_edge(source=self.imagesData[img_key])

                try:
                    self.display_image(source=self.edged_image, widget=self.processedImages[combo_id])
                except TypeError:
                    print("Cannot display Edges of the Image")

            logger.info(f"Viewing {selected_component} of Image {tab_id}")

        # If 2nd tab is selected
        elif tab_id == 1:

            # Get Values from combo box
            # selected_component = self.combo_histogram.currentText().lower()
            selected_component = self.comboBoxes[combo_id].currentText().lower()

            # Histograms Options
            if combo_id == "1_1":
                if selected_component == "original histogram":
                    # Clear old results
                    self.img1_input_histo.clear()
                    self.img1_output_histo.clear()

                    self.output_hist_image = np.copy(self.imagesData[img_key])

                    # Draw the histograms of the input image
                    self.draw_rgb_histogram(source=self.imagesData[img_key], widget=self.img1_input_histo,
                                            title="Original Histogram", label="Pixels")
                    self.draw_rgb_histogram(source=self.imagesData[img_key], widget=self.img1_output_histo,
                                            title="Original Histogram", label="Pixels")

                if selected_component == "equalized histogram":
                    self.img1_output_histo.clear()
                    self.output_hist_image, bins = Histogram.equalize_histogram(source=self.imagesData[img_key],
                                                                                bins_num=255)
                    self.draw_rgb_histogram(source=self.output_hist_image, widget=self.img1_output_histo,
                                            title="Equalized Histogram", label="Pixels")

                elif selected_component == "normalized histogram":
                    self.img1_output_histo.clear()
                    normalized_image, hist, bins = Histogram.normalize_histogram(source=self.imagesData[img_key],
                                                                                 bins_num=255)
                    self.output_hist_image = normalized_image
                    self.draw_rgb_histogram(source=self.output_hist_image, widget=self.img1_output_histo,
                                            title="Normalized Histogram", label="Pixels")

                elif selected_component == "local thresholding":
                    self.img1_output_histo.clear()
                    local_threshold = Histogram.local_threshold(source=self.imagesData[img_key], divs=4)
                    hist, bins = Histogram.histogram(source=local_threshold, bins_num=2)
                    self.output_hist_image = local_threshold
                    self.display_bar_graph(widget=self.img1_output_histo, x=bins, y=hist, width=0.6, brush='r',
                                           title="Local Histogram", label="Pixels")

                elif selected_component == "global thresholding":
                    self.img1_output_histo.clear()
                    global_threshold = Histogram.global_threshold(source=self.imagesData[img_key], threshold=128)
                    hist, bins = Histogram.histogram(source=global_threshold, bins_num=2)
                    self.output_hist_image = global_threshold
                    self.display_bar_graph(widget=self.img1_output_histo, x=bins, y=hist, width=0.6, brush='r',
                                           title="Global Histogram", label="Pixels")

                elif selected_component == "transform to gray":
                    self.img1_output_histo.clear()
                    gray_image = cv2.cvtColor(self.imagesData[img_key], cv2.COLOR_RGB2GRAY)
                    hist, bins = Histogram.histogram(source=gray_image, bins_num=255)
                    self.output_hist_image = gray_image
                    self.display_bar_graph(widget=self.img1_output_histo, x=bins, y=hist, width=0.6, brush='r',
                                           title="Gray Histogram", label="Pixels")

                try:
                    self.display_image(source=self.output_hist_image, widget=self.img1_output)
                except TypeError:
                    print("Cannot display histogram image")

            logger.info(f"Viewing {selected_component} Component Of Image{tab_id}")

        # If Segmentation Tab is selected
        elif tab_id == 7:
            # Get Values from combo box
            selected_component = self.comboBoxes[combo_id].currentText().lower()
            source = np.copy(self.imagesData[combo_id])
            segmented_image = None

            if combo_id == "7_1":
                # TODO: Add QThread for Thresholding Segmentation function

                # Calculate function run time
                start_time = timeit.default_timer()

                regions_x = int(self.text_regions_x_numbers.text())
                regions_y = int(self.text_regions_y_numbers.text())

                # Global Thresholding
                if selected_component == "optimal global thresholding":
                    segmented_image = SegmentationThresholding.apply_optimal_threshold(source=source)

                elif selected_component == "otsu global thresholding":
                    segmented_image = SegmentationThresholding.apply_otsu_threshold(source=source)

                elif selected_component == "spectral global thresholding":
                    segmented_image = SegmentationThresholding.apply_spectral_threshold(source=source)

                # Local Thresholding
                elif selected_component == "optimal local thresholding":
                    segmented_image = SegmentationThresholding.LocalThresholding(
                        source=source, RegionsX=regions_x, RegionsY=regions_y,
                        ThresholdingFunction=SegmentationThresholding.apply_optimal_threshold)

                elif selected_component == "otsu local thresholding":
                    segmented_image = SegmentationThresholding.LocalThresholding(
                        source=source, RegionsX=regions_x, RegionsY=regions_y,
                        ThresholdingFunction=SegmentationThresholding.apply_otsu_threshold)

                elif selected_component == "spectral local thresholding":
                    segmented_image = SegmentationThresholding.LocalThresholding(
                        source=source, RegionsX=regions_x, RegionsY=regions_y,
                        ThresholdingFunction=SegmentationThresholding.apply_spectral_threshold)

                # Function end
                end_time = timeit.default_timer()

                # Show only 5 digits after floating point
                elapsed_time = format(end_time - start_time, '.5f')
                self.label_elapsed_time_thresholding.setText(str(elapsed_time))

            elif combo_id == "7_2":
                # TODO: Add QThread for Clustering Segmentation function

                # Calculate function run time
                start_time = timeit.default_timer()

                # Number of clusters
                k = int(self.text_clusters_numbers.text())
                threshold = int(self.text_clustering_threshold.text())

                if selected_component == "k-means":
                    segmented_image, labels = SegmentationClustering.apply_k_means(source=source, k=k)

                elif selected_component == "region growing":
                    segmented_image = SegmentationClustering.apply_region_growing(source=source)

                elif selected_component == "agglomerative clustering":
                    segmented_image = SegmentationClustering.apply_agglomerative(source=source, clusters_numbers=k,
                                                                                 initial_clusters=25)

                elif selected_component == "mean-shift clustering":
                    segmented_image = SegmentationClustering.apply_mean_shift(source=source, threshold=threshold)

                # Function end
                end_time = timeit.default_timer()

                # Show only 5 digits after floating point
                elapsed_time = format(end_time - start_time, '.5f')
                self.label_elapsed_time_clustering.setText(str(elapsed_time))

            try:
                self.display_image(source=segmented_image, widget=self.processedImages[combo_id])
            except TypeError:
                print("Cannot display Segmented Image")

    def hybrid_image(self):
        """
        Create a hybrid image by applying a high pass filter to one of the images
        and a low pass filter to the other image

        :return: void
        """

        src1 = np.copy(self.imagesData["2_1"])
        src2 = np.copy(self.imagesData["2_2"])

        # Minimum required shape
        min_shape = (min(src1.shape[0], src2.shape[0]),
                     min(src1.shape[1], src2.shape[1]))

        # resize images to ensure both have same shapes
        src1_resized = cv2.resize(src1, min_shape, interpolation=cv2.INTER_AREA)
        src2_resized = cv2.resize(src2, min_shape, interpolation=cv2.INTER_AREA)

        # Apply filters
        image1_dft = FrequencyFilters.high_pass_filter(source=src1_resized, size=20)
        image2_dft = FrequencyFilters.low_pass_filter(source=src2_resized, size=15)

        # Mix 2 images
        hybrid_image = image1_dft + image2_dft

        self.display_image(source=hybrid_image, widget=self.img2_output)

    def hough_transform(self):
        """
        Apply a hough transformation to detect lines or circles in the given image

        :return: void
        """

        hough_image = None

        # Get Parameters Values from the user
        min_radius = int(self.text_min_radius.text())
        max_radius = int(self.text_max_radius.text())
        num_votes = int(self.text_votes.text())

        # Apply the chosen type of hough transform
        if self.radioButton_lines.isChecked():
            hough_image = Hough.hough_lines(source=self.imagesData["3_1"], num_peaks=num_votes)
        elif self.radioButton_circles.isChecked():
            hough_image = Hough.hough_circles(source=self.imagesData["3_1"], min_radius=min_radius,
                                              max_radius=max_radius)

        # Display output
        try:
            self.display_image(source=hough_image, widget=self.img3_output)
        except TypeError:
            print("Cannot display Image")

    def active_contour(self):
        """
        Apply Active Contour Model (Snake) to the given image on a certain shape.
        This algorithm is applied based on Greedy Algorithm
        :return:
        """

        # Get Contour Parameters
        alpha = float(self.text_alpha.text())
        beta = float(self.text_beta.text())
        gamma = float(self.text_gamma.text())
        num_iterations = int(self.text_num_iterations.text())
        num_points_circle = 65
        num_xpoints = 180
        num_ypoints = 180
        w_line = 1
        w_edge = 8

        # Initial variables
        contour_x, contour_y, window_coordinates = None, None, None

        # Calculate function run time
        start_time = timeit.default_timer()

        # Greedy Algorithm

        # copy the image because cv2 will edit the original source in the contour
        image_src = np.copy(self.imagesData["4_1"])

        # Create Initial Contour and display it on the GUI
        if self.radioButton_square_contour.isChecked():
            contour_x, contour_y, window_coordinates = Contour.create_square_contour(source=image_src,
                                                                                     num_xpoints=num_xpoints,
                                                                                     num_ypoints=num_ypoints)
            # Set parameters with pre-tested values for good performance
            alpha = 20
            beta = 0.01
            gamma = 2
            num_iterations = 60
            self.text_alpha.setText(str(alpha))
            self.text_beta.setText(str(beta))
            self.text_gamma.setText(str(gamma))
            self.text_num_iterations.setText(str(num_iterations))

        elif self.radioButton_circle_contour.isChecked():
            contour_x, contour_y, window_coordinates = Contour.create_elipse_contour(source=image_src,
                                                                                     num_points=num_points_circle)
            # Set parameters with pre-tested values for good performance
            alpha = 0.01
            beta = 0.01
            gamma = 2
            num_iterations = 50
            self.text_alpha.setText(str(alpha))
            self.text_beta.setText(str(beta))
            self.text_gamma.setText(str(gamma))
            self.text_num_iterations.setText(str(num_iterations))

        # Display the input image after creating the contour
        src_copy = np.copy(image_src)
        initial_image = self.draw_contour_on_image(src_copy, contour_x, contour_y)
        self.display_image(source=initial_image, widget=self.img4_input)

        # Calculate External Energy which will be used in each iteration of greedy algorithm
        external_energy = gamma * Contour.calculate_external_energy(image_src, w_line, w_edge)

        # Copy the coordinates to update them in the main loop
        cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)

        # main loop of the greedy algorithm
        for iteration in range(num_iterations):
            # Start Applying Active Contour Algorithm
            cont_x, cont_y = Contour.iterate_contour(source=image_src, contour_x=cont_x, contour_y=cont_y,
                                                     external_energy=external_energy,
                                                     window_coordinates=window_coordinates,
                                                     alpha=alpha, beta=beta)

            # Display the new contour after each iteration
            src_copy = np.copy(image_src)
            processed_image = self.draw_contour_on_image(src_copy, cont_x, cont_y)
            self.display_image(source=processed_image, widget=self.img4_output)

            # Used to allow the GUI to update ImageView Object without lagging
            QtWidgets.QApplication.processEvents()

        # Function end
        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
        self.label_snake_time.setText(str(elapsed_time))

    def clear_anchors(self):
        """

        :return:
        """
        print("Clearing anchors")
        self.clear_results(tab_id=self.tab_index)
        self.display_image(source=self.imagesData["4_1"], widget=self.img4_input)

    def reset_contour(self):
        """

        :return:
        """
        print("resetting contour")
        self.clear_results(tab_id=self.tab_index)

    def harris_operator(self):
        """

        :return:
        """
        threshold = float(self.text_harris_threshold.text())
        sensitivity = float(self.text_harris_sensitivity.text())

        # Calculate function run time
        start_time = timeit.default_timer()

        harris_response = Harris.apply_harris_operator(source=self.imagesData["5_1"], k=sensitivity)

        corner_indices, edges_indices, flat_indices = Harris.get_harris_indices(harris_response=harris_response,
                                                                                threshold=threshold)

        img_corners = Harris.map_indices_to_image(source=self.imagesData["5_1"], indices=corner_indices,
                                                  color=[255, 0, 0])

        # Function end
        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
        self.label_harris_time.setText(str(elapsed_time))

        self.display_image(source=img_corners, widget=self.img5_output)

    def sift(self):
        """
        Apply Scale Invariant Feature Detection (SIFT) Algorithm

        :return:
        """

        # Check if both images were loaded
        if ("6_1" not in self.imagesData) or ("6_2" not in self.imagesData):
            self.show_message(header="Warning!!", message="Choose the other image!",
                              button=QMessageBox.Ok, icon=QMessageBox.Warning)
        else:
            img1 = np.copy(self.imagesData["6_1"])
            img2 = np.copy(self.imagesData["6_2"])

            # Calculate function run time
            start_time = timeit.default_timer()

            # Check that user selected a matching method
            if self.combo_matching_methods.currentIndex() == 0:
                self.show_message(header="Warning!!", message="You didn't choose any matching method!",
                                  button=QMessageBox.Ok, icon=QMessageBox.Warning)
            else:
                self.create_sift_thread(source=img1, source_id=0, start_time=start_time)
                self.create_sift_thread(source=img2, source_id=1, start_time=start_time)

    def save_sift_result(self, keypoints: list, descriptors: np.ndarray, source_id: int, elapsed_time: float):
        """
        Save the output from each QThread to use it in the matching
        Then Apply feature matching
        :param keypoints:
        :param descriptors:
        :param source_id:
        :param elapsed_time:
        :return:
        """

        print(f"SIFT Thread {source_id} finished")

        # Update Elapsed Time in GUI Depending on which Thread is finished
        if source_id == 0:
            self.label_sift_A_time.setText(str(elapsed_time))
        elif source_id == 1:
            self.label_sift_B_time.setText(str(elapsed_time))

        QtWidgets.QApplication.processEvents()

        self.sift_results[source_id] = {
            "keypoints": keypoints,
            "descriptors": descriptors
        }

        # Check if 2 threads are finished so we can apply matching
        if (0 in self.sift_results) and (1 in self.sift_results):

            img1 = np.copy(self.imagesData["6_1"])
            img2 = np.copy(self.imagesData["6_2"])
            num_matches = int(self.text_sift_matches.text())

            match_method = None

            # Check which match function to apply
            if self.combo_matching_methods.currentText() == "Sum Square Differences":
                match_method = FeatureMatching.calculate_ssd
            elif self.combo_matching_methods.currentText() == "Normalized Cross Correlations":
                match_method = FeatureMatching.calculate_ncc

            # Calculate function run time
            start_time = timeit.default_timer()

            self.create_matching_thread(source1=img1, source2=img2,
                                        desc1=self.sift_results[0]["descriptors"],
                                        desc2=self.sift_results[1]["descriptors"],
                                        keypoints1=self.sift_results[0]["keypoints"],
                                        keypoints2=self.sift_results[1]["keypoints"],
                                        match_calculator=match_method, num_matches=num_matches,
                                        source_id=3, start_time=start_time)

    def save_filter_result(self, source: np.ndarray, combo_id: str):
        """
        Save the output from Filter QThread to view filter result

        :param source:
        :param combo_id:
        :return:
        """
        print("Filter Processing Finished")
        self.display_image(source=source, widget=self.processedImages[combo_id])

    def save_matching_result(self, source: np.ndarray, source_id: int, elapsed_time: float):
        """
        Save the output from Matching Features QThread

        :param source:
        :param source_id:
        :param elapsed_time:
        :return:
        """

        print(f"Matching Thread {source_id} finished")

        self.label_feature_matching_time.setText(str(elapsed_time))

        # Update Total Time
        max_sift_time = max(float(self.label_sift_A_time.text()), float(self.label_sift_B_time.text()))
        total_time = max_sift_time + float(self.label_feature_matching_time.text())
        self.label_total_matching_time.setText(str(total_time))

        self.display_image(source=source, widget=self.img6_output)

    def save_recognition_result(self, recognized_name: str, elapsed_time: float):
        """

        :param recognized_name:
        :param elapsed_time:
        :return:
        """

        self.label_elapsed_time_face_testing.setText(str(elapsed_time))

        if recognized_name != "Unknown Face!":
            print(f"recognized_name: {recognized_name}")
            self.display_output_face(recognized_name)
        else:
            self.show_message(header="Warning!!", message="Face is not found!!",
                              button=QMessageBox.Ok, icon=QMessageBox.Warning)

    # TODO: Add decorator Function to remove redundancy from Threads
    def create_thread(self, thread_num: int, worker_class: Type[Callable]):
        pass

    def create_sift_thread(self, source: np.ndarray, source_id: int, start_time: float):
        """

        :param source:
        :param source_id:
        :param start_time:
        :return:
        """

        # Step 2: Create a QThread object
        self.threads[source_id] = QThread()

        # Step 3: Create a worker object
        self.workers[source_id] = SIFTWorker(source=source, source_id=source_id, start_time=start_time)

        # Step 4: Move worker to the thread
        self.workers[source_id].moveToThread(self.threads[source_id])

        # Step 5: Connect signals and slots
        self.threads[source_id].started.connect(self.workers[source_id].run)
        self.workers[source_id].finished.connect(self.threads[source_id].quit)
        self.workers[source_id].finished.connect(self.workers[source_id].deleteLater)
        self.threads[source_id].finished.connect(self.threads[source_id].deleteLater)
        self.workers[source_id].finished.connect(self.save_sift_result)
        # self.worker.progress.connect(self.reportProgress)

        # Step 6: Start the thread
        self.threads[source_id].start()

        # Final resets
        self.btn_match_features.setEnabled(False)
        self.threads[source_id].finished.connect(lambda: self.btn_match_features.setEnabled(True))

    def create_matching_thread(self, source1: np.ndarray, source2: np.ndarray,
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
        :return:
        """
        # Step 2: Create a QThread object
        self.threads[source_id] = QThread()

        # Step 3: Create a worker object
        self.workers[source_id] = MatchingWorker(source1=source1, source2=source2, desc1=desc1, desc2=desc2,
                                                 keypoints1=keypoints1, keypoints2=keypoints2,
                                                 match_calculator=match_calculator, num_matches=num_matches,
                                                 source_id=source_id, start_time=start_time)

        # Step 4: Move worker to the thread
        self.workers[source_id].moveToThread(self.threads[source_id])

        # Step 5: Connect signals and slots
        self.threads[source_id].started.connect(self.workers[source_id].run)
        self.workers[source_id].finished.connect(self.threads[source_id].quit)
        self.workers[source_id].finished.connect(self.workers[source_id].deleteLater)
        self.threads[source_id].finished.connect(self.threads[source_id].deleteLater)
        self.workers[source_id].finished.connect(self.save_matching_result)
        # self.worker.progress.connect(self.reportProgress)

        # Step 6: Start the thread
        self.threads[source_id].start()

        # Final resets
        self.btn_match_features.setEnabled(False)
        self.threads[source_id].finished.connect(lambda: self.btn_match_features.setEnabled(True))

    def create_filter_thread(self, source: np.ndarray, source_id: int, filter_function: Callable,
                             shape: int, combo_id: str, sigma: float = 0.0):
        """

        :param source:
        :param source_id:
        :param filter_function:
        :param shape:
        :param combo_id:
        :param sigma:
        :return:
        """

        # Step 2: Create a QThread object
        self.threads[source_id] = QThread()

        # Step 3: Create a worker object
        self.workers[source_id] = FilterWorker(source=source, filter_function=filter_function,
                                               shape=shape, sigma=sigma, combo_id=combo_id)

        # Step 4: Move worker to the thread
        self.workers[source_id].moveToThread(self.threads[source_id])

        # Step 5: Connect signals and slots
        self.threads[source_id].started.connect(self.workers[source_id].run)
        self.workers[source_id].finished.connect(self.threads[source_id].quit)
        self.workers[source_id].finished.connect(self.workers[source_id].deleteLater)
        self.threads[source_id].finished.connect(self.threads[source_id].deleteLater)
        self.workers[source_id].finished.connect(self.save_filter_result)
        # self.worker.progress.connect(self.reportProgress)

        # Step 6: Start the thread
        self.threads[source_id].start()

        # Final resets
        self.combo_filter.setEnabled(False)
        self.threads[source_id].finished.connect(lambda: self.combo_filter.setEnabled(True))

    def create_face_recognition_thread(self, recognizer_obj: object, source_id: int, test_path: str,
                                       start_time: float):
        """

        :param recognizer_obj:
        :param source_id:
        :param test_path:
        :param start_time:
        :return:
        """

        # Step 2: Create a QThread object
        self.threads[source_id] = QThread()

        # Step 3: Create a worker object
        self.workers[source_id] = FaceRecognitionWorker(recognizer_obj=recognizer_obj, test_path=test_path,
                                                        source_id=source_id, start_time=start_time)

        # Step 4: Move worker to the thread
        self.workers[source_id].moveToThread(self.threads[source_id])

        # Step 5: Connect signals and slots
        self.threads[source_id].started.connect(self.workers[source_id].run)
        self.workers[source_id].finished.connect(self.threads[source_id].quit)
        self.workers[source_id].finished.connect(self.workers[source_id].deleteLater)
        self.threads[source_id].finished.connect(self.threads[source_id].deleteLater)
        self.workers[source_id].finished.connect(self.save_recognition_result)
        # self.worker.progress.connect(self.reportProgress)

        # Step 6: Start the thread
        self.threads[source_id].start()

        # Final resets
        self.btn_match_faces.setEnabled(False)
        self.threads[source_id].finished.connect(lambda: self.btn_match_faces.setEnabled(True))

    def detect_faces(self):
        """

        :return:
        """

        thickness = int(self.text_rect_thickness.text())

        # Calculate function run time
        start_time = timeit.default_timer()

        detected_faces = FaceDetection.detect_faces(source=self.imagesData["8_1"])
        faced_image = FaceDetection.draw_faces(source=self.imagesData["8_1"], faces=detected_faces, thickness=thickness)

        # Function end
        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
        self.label_elapsed_time_face_detection.setText(str(elapsed_time))

        self.display_image(source=faced_image, widget=self.img8_1_output)

    def create_database(self):
        """

        :return:
        """

        # Get Database Directory Name
        self.db_path = QtWidgets.QFileDialog.getExistingDirectory(None, str("Choose DB Directory"), os.getcwd(),
                                                                  QtWidgets.QFileDialog.ShowDirsOnly
                                                                  | QtWidgets.QFileDialog.DontResolveSymlinks)

        # Check if user chose a directory
        if self.db_path != "":
            self.recognizer = FaceRecognition.FaceRecognizer(path=self.db_path)
            persons_num, images_num, img_shape = self.recognizer.create_images_matrix()

            # Set Parameters in GUI
            self.label_imgName_9_1.setText(self.db_path.split('/')[-1])
            self.label_imgSize_9_1.setText(f"{img_shape[1]}x{img_shape[0]}")
            self.label_images_num_9_1.setText(str(images_num))
            self.label_persons_num_9_1.setText(str(persons_num))

            self.btn_make_eigenfaces.setEnabled(True)

        # The folder wasn't loaded correctly
        else:
            self.show_message(header="Warning!!", message="You didn't choose any image",
                              button=QMessageBox.Ok, icon=QMessageBox.Warning)

    def make_eigenfaces(self):
        """

        :return:
        """

        # Calculate function run time
        start_time = timeit.default_timer()

        # Train the model
        eigenfaces_num = self.recognizer.fit()
        self.text_eignedfaces_num.setText(str(eigenfaces_num))

        # Function end
        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
        self.label_elapsed_time_face_training.setText(str(elapsed_time))

        # Now User can start matching faces
        self.btn_load_9_2.setEnabled(True)

    def recognize_faces(self):
        """

        :return:
        """

        test_path = self.imagesPaths["9_2"]

        # Calculate function run time
        start_time = timeit.default_timer()

        # TODO: Add QThread For recognize_face Function
        self.create_face_recognition_thread(recognizer_obj=self.recognizer, source_id=3, test_path=test_path,
                                            start_time=start_time)

    def slider_changed(self, indx):
        """
        detects the changes in the sliders using the indx given by ith slider
        and the slider value (add val parameter if you want to use it)
        :param indx: int
        :return: void
        """
        if indx == 0 or indx == 1:
            self.combo_box_changed(tab_id=self.tab_index, combo_id="0_1")

            # Change the filtered image after changing the noisy image
            if (self.img0_noisy.image is not None) and self.combo_filter.currentIndex() != 0:
                self.combo_box_changed(tab_id=self.tab_index, combo_id="0_2")

        elif indx == 2 or indx == 3:
            self.combo_box_changed(tab_id=self.tab_index, combo_id="0_2")

        # Test Images Slider
        elif indx == 4:
            pass

    @staticmethod
    def combine_images(directory_name: str) -> np.ndarray:
        """

        :param directory_name:
        :return:
        """

        images = []
        for file in os.listdir(directory_name):
            # Get Full Path
            file_path = os.path.join(directory_name, file)

            images.append(Image.open(file_path))

        # Remove one image (Display only 9 in GUI)
        images = images[:-1]
        widths, heights = zip(*(image.size for image in images))

        total_width = widths[0] * 3
        max_height = max(heights) * 3

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        y_offset = 0

        for image in images:
            new_im.paste(image, (x_offset, y_offset))

            if x_offset != (total_width - image.size[0]):
                x_offset += image.size[0]
            else:
                x_offset = 0
                y_offset += image.size[1]

        return new_im

    def display_output_face(self, class_name: str):
        """

        :param class_name:
        :return:
        """

        path = self.db_path + class_name
        recognized_face = self.combine_images(directory_name=path)
        self.display_image(source=recognized_face, widget=self.img9_2_output)

    @staticmethod
    def draw_contour_on_image(source, points_x, points_y):
        """
        This function draws a given contour coordinates on the given image

        :param source: image source to draw the contour above it
        :param points_x: list of indices of the contour in x-direction
        :param points_y: list of indices of the contour in y-direction
        :return:
        """

        # Copy the image source to prevent modifying the original image
        src = np.copy(source)

        points = []
        for px, py in zip(points_x, points_y):
            points.append([px, py])

        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))

        image = cv2.polylines(src, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return image

    @staticmethod
    def display_image(source: np.ndarray, widget: pg.ImageView):
        """
        Displays the given image source in the specified ImageView widget

        :param source: image source
        :param widget: ImageView object
        :return: void
        """

        # Copy the original source because cv2 updates the passed parameter
        src = np.copy(source)

        # Rotate the image 90 degree because ImageView is rotated
        src = cv2.transpose(src)

        widget.setImage(src)
        widget.view.setRange(xRange=[0, src.shape[0]], yRange=[0, src.shape[1]],
                             padding=0)
        widget.ui.roiPlot.hide()

    @staticmethod
    def display_bar_graph(widget, x, y, width, brush, title, label):
        """

        :param widget: widget object to draw the graph on it
        :param x: list of numbers in the x-direction
        :param y: list of numbers in the y-direction
        :param width: width of the bar
        :param brush: color of the bar
        :param title: title of the window
        :param label: bottom label of the window
        :return:
        """

        # Create BarGraphItem and add it to the widget
        bg = pg.BarGraphItem(x=x, height=y, width=width, brush=brush)
        widget.addItem(bg)

        # Adjust Widget
        widget.plotItem.setTitle(title)
        widget.plotItem.showGrid(True, True, alpha=0.8)
        widget.plotItem.setLabel("bottom", text=label)

        # Auto scale Y Axis
        vb = widget.getViewBox()
        vb.setAspectLocked(lock=False)
        vb.setAutoVisible(y=1.0)
        vb.enableAutoRange(axis='y', enable=True)

    @staticmethod
    def clear_bar_graph(widget):
        """

        :param widget:
        :return:
        """
        # Auto scale Y Axis
        vb = widget.getViewBox()
        vb.setAspectLocked(lock=False)
        vb.setAutoVisible(y=1.0)
        vb.enableAutoRange(axis='y', enable=True)

    @staticmethod
    def draw_rgb_histogram(source: np.ndarray, widget, title: str = "title", label: str = "label"):
        """

        :param source: image source
        :param widget: widget object to draw the histogram in
        :param title: title of the window
        :param label: bottom label of the window
        :return:
        """

        # Create pens list with red, green and blue
        pens = [pg.mkPen(color=(255, 0, 0)), pg.mkPen(color=(0, 255, 0)),
                pg.mkPen(color=(0, 0, 255))]

        # Adjust Widget
        widget.plotItem.setTitle(title)
        widget.plotItem.showGrid(True, True, alpha=0.8)
        widget.plotItem.setLabel("bottom", text=label)

        for i in range(source.shape[2]):
            hist, bins = Histogram.histogram(source=source[:, :, i], bins_num=255)

            # setting pen=(i,3) automatically creates three different-colored pens
            widget.plot(bins, hist[:-1], pen=pens[i])

    @staticmethod
    def draw_gray_histogram(source: np.ndarray, widget, bins_num):
        """

        :param source: image source
        :param widget: widget object to draw the graph on it
        :param bins_num: number of bins to split the histogram into
        :return:
        """

        # Create histogram and plot it
        hist, bins = Histogram.histogram(source=source, bins_num=bins_num)
        widget.plot(bins, hist)

    @staticmethod
    def show_message(header, message, button, icon):
        """
        Show a pop-up window with a message
        :param header: message header of the pop-up window
        :param message: main message content
        :param button: button type
        :param icon: icon type
        :return:
        """

        msg = QMessageBox()
        msg.setWindowTitle(header)
        msg.setText(message)
        msg.setIcon(icon)
        msg.setStandardButtons(button)
        msg.exec_()


def main():
    """
    the application startup functions
    :return:
    """

    app = QtWidgets.QApplication(sys.argv)

    # set stylesheet
    file = QFile("UI/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    main_window = QtWidgets.QMainWindow()
    ImageProcessor(main_window)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
