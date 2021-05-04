# Importing Packages
# importing module
import logging
import sys

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.QtWidgets import QMessageBox

from UI import mainGUI as m
from libs import EdgeDetection, Noise, LowPass, Histogram, FrequencyFilters, \
                 Hough, Contour, Harris, SIFT, FeatureMatching

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(lineno)s - %(levelname)s - %(message)s',
                    filemode='w')

# Creating an object
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
                            [self.img7_1_input, self.img7_2_input]]

        self.outputImages = [[self.img0_noisy, self.img0_filtered, self.img0_edged],
                             self.img1_output, self.img2_output, self.img3_output,
                             self.img4_output, self.img5_output, self.img6_output,
                             [self.img7_1_output, self.img7_2_output]]

        self.filtersImages = [self.img0_noisy, self.img0_filtered, self.img0_edged]

        self.histoImages = [self.img1_input_histo, self.img1_output, self.img1_output_histo]

        # This contains all the widgets to setup them in one loop
        self.imageWidgets = [self.img0_input, self.img0_noisy, self.img0_filtered, self.img0_edged,
                             self.img1_input, self.img1_output,
                             self.img2_1_input, self.img2_2_input, self.img2_output,
                             self.img3_input, self.img3_output,
                             self.img4_input, self.img4_output,
                             self.img5_input, self.img5_output,
                             self.img6_1_input, self.img6_2_input, self.img6_output,
                             self.img7_1_input, self.img7_2_input, self.img7_1_output, self.img7_2_output]

        # Initial Variables
        self.currentNoiseImage = None
        self.edged_image = None
        self.filtered_image = None
        self.output_hist_image = None
        self.updated_image = None

        # Dictionaries to store images data
        self.imagesData = {}
        self.heights = {}
        self.weights = {}

        # Images Labels and Sizes
        self.imagesLabels = {"0_1": self.label_imgName_0,   "1_1": self.label_imgName_1,
                             "2_1": self.label_imgName_2_1, "2_2": self.label_imgName_2_2,
                             "3_1": self.label_imgName_3,   "4_1": self.label_imgName_4,
                             "5_1": self.label_imgName_5,
                             "6_1": self.label_imgName_6_1, "6_2": self.label_imgName_6_2,
                             "7_1": self.label_imgName_7_1, "7_2": self.label_imgName_7_2}

        self.imagesSizes = {"0_1": self.label_imgSize_0,   "1_1": self.label_imgSize_1,
                            "2_1": self.label_imgSize_2_1, "2_2": self.label_imgSize_2_2,
                            "3_1": self.label_imgSize_3,   "4_1": self.label_imgSize_4,
                            "5_1": self.label_imgSize_5,
                            "6_1": self.label_imgSize_6_1, "6_2": self.label_imgSize_6_2,
                            "7_1": self.label_imgSize_7_1, "7_2": self.label_imgSize_7_2}

        # list contains the last pressed values
        self.sliderValuesClicked = {0: ..., 1: ..., 2: ..., 3: ...}
        self.sliders = [self.snr_slider_1, self.sigma_slider_1, self.mask_size_1, self.sigma_slider_2]

        # Sliders Connections
        for slider in self.sliders:
            slider.id = self.sliders.index(slider)
            slider.signal.connect(self.slider_changed)

        # Combo Lists
        self.updateCombos = [self.combo_noise, self.combo_filter, self.combo_edges, self.combo_histogram]

        # Setup Load Buttons Connections
        self.btn_load_0.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2_2.clicked.connect(lambda: self.load_file(self.tab_index, True))
        self.btn_load_3.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_4.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_5.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_6_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_6_2.clicked.connect(lambda: self.load_file(self.tab_index, True))
        self.btn_load_7_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_7_2.clicked.connect(lambda: self.load_file(self.tab_index, True))

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.combo_box_changed(self.tab_index, 0))
        self.combo_filter.activated.connect(lambda: self.combo_box_changed(self.tab_index, 1))
        self.combo_edges.activated.connect(lambda: self.combo_box_changed(self.tab_index, 2))
        self.combo_histogram.activated.connect(lambda: self.combo_box_changed(self.tab_index, 3))

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
        self.btn_match_sift.clicked.connect(self.sift)

        self.setup_images_view()

    def tab_changed(self):
        """
        Updates the current tab index

        :return: void
        """

        self.tab_index = self.Main_TabWidget.currentIndex()
        print(f"Current Tab index: {self.tab_index}")

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
            self.heights[img_idx], self.weights[img_idx], _ = img_rgb.shape

            # Reset Results
            self.clear_results(tab_id=img_id)

            # Display Original Image
            self.display_image(source=self.imagesData[img_idx], widget=self.inputImages[img_id][multi_widget])

            # Enable the comboBoxes and settings
            self.enable_gui(tab_id=img_id)

            # Set Image Name and Sizes
            self.imagesLabels[img_idx].setText(img_name)
            self.imagesSizes[img_idx].setText(f"{self.heights[img_idx]}x"
                                              f"{self.weights[img_idx]}")

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

        if tab_id == 0:
            for i in range(len(self.updateCombos)):
                # Enable Combo Boxes
                self.updateCombos[i].setEnabled(True)

        elif tab_id == 1:
            self.combo_histogram.setEnabled(True)

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
            self.sift_settings_layout.setEnabled(True)
            self.btn_load_6_2.setEnabled(True)
            self.btn_match_sift.setEnabled(True)

        # in Template Matching  Tab
        elif tab_id == 7:
            self.template_matching_settings_layout.setEnabled(True)
            self.btn_match_template.setEnabled(True)

    def clear_results(self, tab_id):
        """
        Clears previous results when loading a new image

        :param tab_id: current tab index
        :return: void
        """

        # Check current tab index
        if tab_id == 0:
            # Clear Filters Images Widgets
            for i in range(len(self.filtersImages)):
                self.filtersImages[i].clear()

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
            for widget in self.histoImages:
                widget.clear()

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
            self.img7_1_input.clear()
            self.img7_1_output.clear()
            self.img7_2_output.clear()

    def combo_box_changed(self, tab_id, combo_id):
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
            selected_component = self.updateCombos[combo_id].currentText().lower()

            # Adjust Sliders Values
            noise_snr = self.snr_slider_1.value() / 10
            noise_sigma = self.sigma_slider_1.value()  # This value from 0 -> 4 so we need to map to another range
            noise_sigma = np.round(np.interp(noise_sigma, [0, 4], [0, 255]))  # This value from 0 -> 255

            filter_sigma = self.sigma_slider_2.value()
            filter_sigma = np.round(np.interp(filter_sigma, [0, 4], [0, 255]))

            mask_size = self.mask_size_1.value()
            mask_size = int(np.round(np.interp(mask_size, [1, 4], [3, 9])))

            # Noise Options
            if combo_id == 0:
                self.snr_slider_1.setEnabled(True)

                if selected_component == "uniform noise":
                    self.currentNoiseImage = Noise.uniform_noise(source=self.imagesData[img_key], snr=noise_snr)

                elif selected_component == "gaussian noise":
                    self.sigma_slider_1.setEnabled(True)
                    self.currentNoiseImage = Noise.gaussian_noise(source=self.imagesData[img_key], sigma=noise_sigma,
                                                                  snr=noise_snr)

                elif selected_component == "salt & pepper noise":
                    self.currentNoiseImage = Noise.salt_pepper_noise(source=self.imagesData[img_key], snr=noise_snr)

                try:
                    self.display_image(source=self.currentNoiseImage, widget=self.filtersImages[combo_id])
                except TypeError:
                    print("Cannot display Image")

            # Filters Options
            if combo_id == 1:
                self.mask_size_1.setEnabled(True)

                # Check if there's a noisy image already
                if self.currentNoiseImage is None:
                    self.show_message(header="Warning!!", message="Apply noise to the image first",
                                      button=QMessageBox.Ok, icon=QMessageBox.Warning)

                elif selected_component == "average filter":
                    self.filtered_image = LowPass.average_filter(source=self.currentNoiseImage, shape=mask_size)

                elif selected_component == "gaussian filter":
                    self.sigma_slider_2.setEnabled(True)
                    self.filtered_image = LowPass.gaussian_filter(source=self.currentNoiseImage, shape=mask_size,
                                                                  sigma=filter_sigma)

                elif selected_component == "median filter":
                    self.filtered_image = LowPass.median_filter(source=self.currentNoiseImage, shape=mask_size)

                try:
                    self.display_image(source=self.filtered_image, widget=self.filtersImages[combo_id])
                except TypeError:
                    print("Cannot display Image")

            # Edge Detection Options
            if combo_id == 2:
                if selected_component == "sobel mask":
                    self.edged_image = EdgeDetection.sobel_edge(source=self.imagesData[img_key])

                elif selected_component == "roberts mask":
                    self.edged_image = EdgeDetection.roberts_edge(source=self.imagesData[img_key])

                elif selected_component == "prewitt mask":
                    self.edged_image = EdgeDetection.prewitt_edge(source=self.imagesData[img_key])

                elif selected_component == "canny mask":
                    self.edged_image = EdgeDetection.canny_edge(source=self.imagesData[img_key])

                try:
                    self.display_image(source=self.edged_image, widget=self.filtersImages[combo_id])
                except TypeError:
                    print("Cannot display Image")

            logger.info(f"Viewing {selected_component} Of Image #{tab_id}")

        # If 2nd tab is selected
        elif tab_id == 1:

            # Get Values from combo box
            selected_component = self.combo_histogram.currentText().lower()

            # Histograms Options
            if combo_id == 3:
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

            logger.info(f"Viewing {selected_component} Component Of Image #{tab_id}")

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
        print("Applying harris operator")
        threshold = float(self.text_harris_threshold.text())
        sensitivity = float(self.text_harris_sensitivity.text())
        # window_size = int(self.text_harris_window_size.text())

        harris_response = Harris.apply_harris_operator(source=self.imagesData["5_1"], k=sensitivity)

        # harris_response = Harris.apply_harris_operator(source=self.imagesData["5_1"], k=sensitivity,
        #                                                 window_size=window_size)

        corner_indices, edges_indices, flat_indices = Harris.get_harris_indices(harris_response=harris_response,
                                                                                threshold=threshold)

        img_corners = Harris.map_indices_to_image(source=self.imagesData["5_1"], indices=corner_indices,
                                                  color=[255, 0, 0])

        self.display_image(source=img_corners, widget=self.img5_output)

    def sift(self):
        """

        :return:
        """
        print("Applying SIFT Matching")
        img1 = np.copy(self.imagesData["6_1"])
        img2 = np.copy(self.imagesData["6_2"])

        img1_copy = np.copy(self.imagesData["6_1"])
        img2_copy = np.copy(self.imagesData["6_2"])

        num_matches = int(self.text_sift_matches.text())

        keypoints_1, descriptors_1 = SIFT.Sift(source=img1)
        keypoints_2, descriptors_2 = SIFT.Sift(source=img2)

        matches = FeatureMatching.apply_feature_matching(descriptors_1, descriptors_2, FeatureMatching.calculate_ncc)
        matches = sorted(matches, key=lambda x: x.distance, reverse=True)

        matched_image = cv2.drawMatches(img1_copy, keypoints_1, img2_copy, keypoints_2,
                                        matches[:num_matches], img2_copy, flags=2)

        self.display_image(source=matched_image, widget=self.img6_output)

    def slider_changed(self, indx):
        """
        detects the changes in the sliders using the indx given by ith slider
        and the slider value
        :param indx: int
        :return: none
        """
        if indx == 0 or indx == 1:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=0)

            # Change the filtered image after changing the noisy image
            if self.currentNoiseImage is not None:
                self.combo_box_changed(tab_id=self.tab_index, combo_id=1)

        elif indx == 2 or indx == 3:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=1)

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
