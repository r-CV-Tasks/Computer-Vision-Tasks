# Importing Packages
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
import pyqtgraph as pg
from threading import Thread
from UI import mainGUI as m
from libs.helpers import map_ranges
from libs.imageModel import *

# importing module
import logging

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

        # Setup Tab Widget Connections
        self.tabWidget_process.setCurrentIndex(0)
        self.tab_index = self.tabWidget_process.currentIndex()
        self.tabWidget_process.currentChanged.connect(self.tab_changed)

        # Images Lists
        self.inputImages = [self.img1_input, self.img2_input, self.imgA_input, self.imgB_input]
        self.filtersImages = [self.img1_noisy, self.img1_filtered, self.img1_edged]
        self.histoImages = [self.img2_input_histo, self.img2_output, self.img2_output_histo]

        self.imageWidgets = [self.img1_input, self.img1_noisy, self.img1_filtered, self.img1_edged,
                             self.img2_input, self.img2_output,
                             self.imgA_input, self.imgB_input, self.imgX_output]

        # No Noisy Image Array yet
        self.currentNoiseImage = None

        self.imagesData = {1: ..., 2: ..., 3: ..., 4: ...}
        self.heights = [..., ..., ..., ...]
        self.weights = [..., ..., ..., ...]

        # Images Labels and Sizes
        self.imagesLabels = {1: [self.label_imgName_1], 2: [self.label_imgName_2],
                             3: [self.label_imgName_3], 4: [self.label_imgName_4]}

        self.imagesSizes = {1: [self.label_imgSize_1], 2: [self.label_imgSize_2],
                            3: [self.label_imgSize_3], 4: [self.label_imgSize_4]}

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
        self.btn_load_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_3.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_4.clicked.connect(lambda: self.load_file(self.tab_index + 1))

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.combo_box_changed(self.tab_index, 0))
        self.combo_filter.activated.connect(lambda: self.combo_box_changed(self.tab_index, 1))
        self.combo_edges.activated.connect(lambda: self.combo_box_changed(self.tab_index, 2))
        self.combo_histogram.activated.connect(lambda: self.combo_box_changed(self.tab_index, 3))

        # Setup Hybrid Button
        self.btn_hybrid.clicked.connect(self.hybrid_image)

        self.setup_images_view()

    def tab_changed(self):
        """
        Updates the current tab index
        :return:
        """
        self.tab_index = self.tabWidget_process.currentIndex()

    def setup_images_view(self):
        """
        Adjust the shape and scales of the widgets
        Remove unnecessary options
        :return:
        """
        for widget in self.imageWidgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def load_file(self, img_id):
        """
        Load the File from User
        :param img_id: 0, 1, 2 or 3
        :return:
        """

        # Open File & Check if it was loaded correctly
        logger.info("Browsing the files...")
        repo_path = "./src/Images"
        self.filename, self.format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                           "*;;" "*.jpg;;" "*.jpeg;;" "*.png;;")
        img_name = self.filename.split('/')[-1]
        if self.filename == "":
            pass
        else:
            image = cv2.imread(self.filename, flags=cv2.IMREAD_GRAYSCALE).T
            self.heights[img_id], self.weights[img_id] = image.shape

            bgr_img = cv2.imread(self.filename)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            imgByte_RGB = cv2.transpose(rgb_img)
            self.imagesData[img_id] = imgByte_RGB

            # When Images in Tab1, Tab2 or Image A in Tab 3
            if img_id != 3:
                # Reset Results
                self.clear_results(tab_id=img_id)

                # Clear imgX output when uploading a new image
                if self.tab_index == 2:
                    self.clear_results(tab_id=self.tab_index)

                # Create and Display Original Image
                self.display_image(data=self.imagesData[img_id], widget=self.inputImages[img_id])

                # Enable the combo box and parameters input
                self.enable_gui(tab_id=img_id)

                # Set Image Name and Size
                self.imagesLabels[img_id + 1][0].setText(img_name)
                self.imagesSizes[img_id + 1][0].setText(
                    f"{self.imagesData[img_id].shape[0]}x{self.imagesData[img_id].shape[1]}")

                logger.info(f"Added Image{img_id + 1}: {img_name} successfully")

            # When Loading Image B in Tab 3
            else:
                if self.heights[3] != self.heights[2] or self.weights[3] != self.weights[2]:
                    self.show_message("Warning!!", "Images sizes must be the same, please upload another image",
                                      QMessageBox.Ok, QMessageBox.Warning)
                    logger.warning("Warning!!. Images sizes must be the same, please upload another image")
                else:
                    # Reset Results
                    self.clear_results(tab_id=img_id)

                    self.display_image(self.imagesData[img_id], self.inputImages[img_id])

                    # Set Image Name and Size
                    self.imagesLabels[img_id + 1][0].setText(img_name)
                    self.imagesSizes[img_id + 1][0].setText(
                        f"{self.imagesData[img_id].shape[0]}x{self.imagesData[img_id].shape[1]}")
                    self.btn_hybrid.setEnabled(True)
                    logger.info(f"Added Image{img_id + 1}: {img_name} successfully")

    def enable_gui(self, tab_id):
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
            if type(self.imagesData[3]) != type(...):
                self.btn_hybrid.setEnabled(True)

    def clear_results(self, tab_id):
        # Reset previous outputs
        if tab_id == 0:
            # Clear Images Widgets
            for i in range(len(self.filtersImages)):
                self.filtersImages[i].clear()
        elif tab_id == 1:
            for widget in self.histoImages:
                widget.clear()

        elif tab_id == 2:
            self.imgX_output.clear()

    def combo_box_changed(self, tab_id, combo_id):
        """

        :param tab_id:
        :param combo_id:
        :return:
        """

        # If 1st tab is selected
        if tab_id == 0:
            # Get Values from combo box and sliders
            selected_component = self.updateCombos[combo_id].currentText().lower()

            # Adjust Sliders Values
            noise_snr = self.snr_slider_1.value() / 10
            noise_sigma = self.sigma_slider_1.value()  # This value from 0 -> 4
            noise_sigma = np.round(map_ranges(noise_sigma, 0, 4, 0, 255))  # This value from 0 -> 255

            filter_sigma = self.sigma_slider_2.value()
            filter_sigma = np.round(map_ranges(filter_sigma, 0, 4, 0, 255))

            mask_size = self.mask_size_1.value()
            mask_size = int(np.round(map_ranges(mask_size, 1, 4, 3, 9)))

            # Noise Options
            if combo_id == 0:
                self.snr_slider_1.setEnabled(True)
                if selected_component == "uniform noise":
                    self.currentNoiseImage = add_noise(data=self.imagesData[0], type="uniform", snr=noise_snr)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])

                elif selected_component == "gaussian noise":
                    self.sigma_slider_1.setEnabled(True)
                    self.currentNoiseImage = add_noise(data=self.imagesData[0], type="gaussian", snr=noise_snr,
                                                       sigma=noise_sigma)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])

                elif selected_component == "salt & pepper noise":
                    self.currentNoiseImage = add_noise(data=self.imagesData[0], type="salt & pepper", snr=noise_snr)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])

            # Filters Options
            if combo_id == 1:
                self.mask_size_1.setEnabled(True)

                # Check if there's a noisy image already
                if self.currentNoiseImage is None:
                    self.show_message(header="Warning!!", message="Apply noise to the image first",
                                      button=QMessageBox.Ok, icon=QMessageBox.Warning)

                elif selected_component == "average filter":
                    filtered_image = apply_filter(data=self.currentNoiseImage, type="average", shape=mask_size)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])

                elif selected_component == "gaussian filter":
                    self.sigma_slider_2.setEnabled(True)
                    filtered_image = apply_filter(data=self.currentNoiseImage, type="gaussian", shape=mask_size,
                                                  sigma=filter_sigma)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])

                elif selected_component == "median filter":
                    filtered_image = apply_filter(data=self.currentNoiseImage, type="median", shape=mask_size)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])

            # Edge Detection Options
            if combo_id == 2:
                if selected_component == "sobel mask":
                    edged_image = apply_edge_mask(data=self.imagesData[0], type="sobel")
                    self.display_image(data=edged_image, widget=self.filtersImages[combo_id])

                elif selected_component == "roberts mask":
                    edged_image = apply_edge_mask(data=self.imagesData[0], type="roberts")
                    self.display_image(data=edged_image, widget=self.filtersImages[combo_id])

                elif selected_component == "prewitt mask":
                    edged_image = apply_edge_mask(data=self.imagesData[0], type="prewitt")
                    self.display_image(data=edged_image, widget=self.filtersImages[combo_id])

                elif selected_component == "canny mask":
                    edged_image = apply_edge_mask(data=self.imagesData[0], type="canny")
                    self.display_image(data=edged_image, widget=self.filtersImages[combo_id])

            logger.info(f"Viewing {selected_component} Component Of Image{combo_id + 1}")

        # If 2nd tab is selected
        elif tab_id == 1:
            selected_component = self.combo_histogram.currentText().lower()

            # Histograms Options
            if combo_id == 3:
                if selected_component == "original histogram":
                    self.img2_input_histo.clear()
                    self.draw_rgb_histogram(data=self.imagesData[1], widget=self.img2_input_histo)

                if selected_component == "equalized histogram":
                    self.img2_output_histo.clear()
                    histo, bins = get_histogram(data=self.imagesData[1], type="equalized", bins_num=255)
                    self.draw_rgb_histogram(data=histo, widget=self.img2_output_histo)
                    self.display_image(data=histo, widget=self.img2_output)

                elif selected_component == "normalized histogram":
                    self.img2_output_histo.clear()
                    normalized_image, histo, bins = get_histogram(data=self.imagesData[1], type="normalized", bins_num=255)
                    self.draw_rgb_histogram(data=normalized_image, widget=self.img2_output_histo)
                    self.display_image(data=normalized_image, widget=self.img2_output)

                elif selected_component == "local thresholding":
                    self.img2_output_histo.clear()
                    local_threshold = thresholding(data=self.imagesData[1], type="local", threshold=128, divs=4)
                    histo, bins = get_histogram(data=local_threshold, type="original", bins_num=2)
                    self.display_bar_graph(widget=self.img2_output_histo, x=bins, y=histo, width=0.6, brush='r',
                                           title="Local Histogram", label="Pixels")

                    self.display_image(data=local_threshold, widget=self.img2_output)

                elif selected_component == "global thresholding":
                    self.img2_output_histo.clear()
                    global_threshold = thresholding(data=self.imagesData[1], type="global", threshold=128)
                    histo, bins = get_histogram(data=global_threshold, type="original", bins_num=2)
                    self.display_bar_graph(widget=self.img2_output_histo, x=bins, y=histo, width=0.6, brush='r',
                                           title="Global Histogram", label="Pixels")

                    self.display_image(data=global_threshold, widget=self.img2_output)

                elif selected_component == "transform to gray":
                    self.img2_output_histo.clear()
                    gray_image = rgb_to_gray(data=self.imagesData[1])
                    self.display_image(data=gray_image, widget=self.img2_output)
                    histo, bins = get_histogram(data=gray_image, type="original", bins_num=255)
                    self.display_bar_graph(widget=self.img2_output_histo, x=bins, y=histo, width=0.6, brush='r',
                                           title="Gray Histogram", label="Pixels")

            logger.info(f"Viewing {selected_component} Component Of Image{combo_id + 1}")

    def hybrid_image(self):
        """

        :return:
        """
        self.hybrid_image = mix_images(data1=self.imagesData[2], data2=self.imagesData[3], hpf_size=20, lpf_size=15)
        self.display_image(widget=self.imgX_output, data=self.hybrid_image)

    def slider_changed(self, indx, val):
        """
        detects the changes in the sliders using the indx given by ith slider
        and the slider value
        :param indx: int
        :param val: int
        :return: none
        """
        print(f"Slider {indx} With Value {val}")
        # self.sliderValuesClicked[indx] = val / 10
        if indx == 0 or indx == 1:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=0)

            # Change the filtered image after changing the noisy image
            if self.currentNoiseImage is not None:
                self.combo_box_changed(tab_id=self.tab_index, combo_id=1)

        elif indx == 2 or indx == 3:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=1)

    def display_image(self, data, widget):
        """
        Display the given data
        :param data: 2d numpy array
        :param widget: ImageView object
        :return:
        """
        widget.setImage(data)
        widget.view.setRange(xRange=[0, data.shape[0]], yRange=[0, data.shape[1]],
                             padding=0)
        widget.ui.roiPlot.hide()

    def display_bar_graph(self, widget, x, y, width, brush, title, label):
        """

        :param x:
        :param y:
        :param width:
        :param brush:
        :param widget:
        :return:
        """

        # Create BarGraphItem and add it to the widget
        bg = pg.BarGraphItem(x=x, height=y, width=width, brush=brush)
        widget.addItem(bg)

        widget.plotItem.setTitle(title)
        widget.plotItem.showGrid(True, True, alpha=0.8)
        widget.plotItem.setLabel("bottom", text=label)

        # Auto scale Y Axis
        vb = widget.getViewBox()
        vb.setAspectLocked(lock=False)
        vb.setAutoVisible(y=1.0)
        vb.enableAutoRange(axis='y', enable=True)

    def draw_rgb_histogram(self, data: np.ndarray, widget):
        """

        :param data:
        :param widget:
        :return:
        """

        pens = [pg.mkPen(color=(255, 0, 0)), pg.mkPen(color=(0, 255, 0)),
                pg.mkPen(color=(0, 0, 255))]

        for i in range(data.shape[2]):
            y, x = get_histogram(data=data[:, :, i], type="original", bins_num=255)

            # setting pen=(i,3) automatically creates three different-colored pens
            widget.plot(x, y[:-1], pen=pens[i])

    def draw_gray_histogram(self, data: np.ndarray, widget, bins_num):
        """

        :param data:
        :param widget:
        :param bins_num:
        :return:
        """
        y, x = get_histogram(data=data, type="original", bins_num=bins_num)
        widget.plot(x, y)

    @staticmethod
    def show_message(header, message, button, icon):
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
    main_window = QtWidgets.QMainWindow()
    ImageProcessor(main_window)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
