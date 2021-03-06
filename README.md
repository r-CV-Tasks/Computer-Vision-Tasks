<h1 style="text-align: center;"> Computer Vision Course Tasks</h1>

In this Repository we present a variety of Image Processing Techniques implemented from scratch using `Python` with help of some helpful packages.

## Table of contents
### <a href="#installation_h">Installation</a>

### <a href="#usage_h">Usage</a>

### <a href="#image-processing_h">Image Processing</a>
* [Adding Noise To Image](#1-adding-noise-to-image)
* [Image Filtering](#2-image-filtering)
* [Edge Detection](#3-edge-detection)
* [Image Histogram and Thresholding](#4-image-histogram-and-thresholding)
* [Hybrid Images](#5-hybrid-images)


### <a href="#boundary-detection_h">Boundary Detection</a>
* [Hough Transformation (Lines and Circles Detection)](#1-hough-transformation)
* [Active Contour Model (Snake)](#2-active-contour-model)

### <a href="#features-detection-and-image-matching_h">Features Detection and Image Matching</a>
* [Feature Extraction In Images Using Harris Operator](#1-extract-the-unique-features-in-all-images-using-harris-operator)

* [Feature Descriptors Using Scale Invariant Features (SIFT) Algorithm](#2-feature-descriptors-using-scale-invariant-features-sift-algorithm)

* [Matching the Image Set Features](#3-matching-the-image-set-features)

### <a href="#image-segmentation_h">Image Segmentation</a>
* [Using Thresholding Techniques](#1-thresholding-segmentation)
* [Using Clustering Methods](#2-clustering-segmentation)

### <a href="#face-detection-and-recognition_h">Face Detection and Recognition</a>
* [Face Detection (Color or Gray-scale)](#1-face-detection)
* [Face Recognition (Based on PCA/Eigen Analysis)](#2-face-recognition)

<div style="page-break-after: always;"></div>


# <a name="installation_h">Installation</a>

To install the required libraries and dependencies, open your terminal in the repository directory and run this command:
```
pip install -r requirements.txt
```

### Script's Components
requirements.txt contains the versions of each libraries, if already installed the installation will be skipped:
- Scipy
- Numpy
- Pyqtgraph
- PyQt5
- opencv-python
- Pillows
- Matplotlib
- scikit_learn

# <a name="usage_h">Usage</a>
The **GUI** is composed of many tabs; each tab contains some push buttons, combo boxes or sliders, input texts and some widgets to view the images.

Each category of implemented Algorithms is displayed in a separate tab in the GUI.

Simply you could load the image you want to apply the algorithm on via push buttons, adjust the required parameters then apply the selected algorithm.

Here's the view of the UI tabs without loading any images or applying any algorithms.

<details>
  <h4><summary>Main UI Tabs</summary></h4>
  
  <img src="resources/UI/Noise_Filters_Edges_Tab.png" alt="Noise_Filters_Edges_Tab"
  name="Noise_Filters_Edges_Tab" target="_blank" width="500" height="400">
  
  <img src="resources/UI/Histogram_Tab.png" alt="Histogram_Tab" 
  name="Histogram_Tab" width="500" height="400">
  
  <img src="resources/UI/Hybrid_Tab.png" alt="Hybrid_Tab" 
  name="Hybrid_Tab" width="500" height="400">
  
  <img src="resources/UI/Hough_Tab.png" alt="Hough_Tab.png"
  name="Hough_Tab" width="500" height="400">
  
  <img src="resources/UI/Active_Contour_Tab.png" alt="Active_Contour_Tab" 
  name="Active_Contour_Tab" width="500" height="400">
  
  <img src="resources/UI/Harris_Tab.png" alt="Harris_Tab" 
  name="Harris_Tab" width="500" height="400">
  
  <img src="resources/UI/SIFT_Tab.png" alt="SIFT_Tab" 
  name="SIFT_Tab" width="500" height="400">
  
  <img src="resources/UI/Segmentation_Tab.png" alt="Segmentation_Tab" 
  name="Segmentation_Tab" width="500" height="400">
</details>

<div style="page-break-after: always;"></div>

<!-- Task #1 Report -->

# <a name="image-processing_h">Image Processing</a>
In this section we present some implementations such as adding noise to image, filtering the added noise, viewing different types of histograms, applying threshold to image and hybrid images.

## 1. Adding Noise To Image
We implemented 3 types of noise: `Uniform`, `Gaussian` and `Salt & Pepper`. In each type, you could adjust some parameters such as **Signal-To-Noise Ratio (SNR)** and **Sigma** to show different outputs.

The results below were taken with the following setup:

**Noise parameters:**
- `SNR` = 0.6
- `Sigma` = 128 (For Gaussian Noise Only)

The whole GUI is displayed to show you the difference between the original and the noisy image.

### 1.1 Uniform Noise
<img src="resources/results/image_processing/Noise_Uniform_1.png" alt="Noise_Uniform_1" width="600" height="500">

### 1.2 Gaussian Noise
<img src="resources/results/image_processing/Noise_Gaussian_1.png" alt="Noise_Gaussian_1" width="600" height="500">

### 1.3 Salt & Pepper Noise
<img src="resources/results/image_processing/Noise_Salt_and_Pepper_1.png" alt="Noise_Salt_and_Pepper_1" width="600" height="500">

To decrease amount of noise, move the SNR slider a little, and this would be the new output with `SNR = 0.9`, which means only 10% of the image is noise.

<img src="resources/results/image_processing/Noise_Salt_and_Pepper_2.png" alt="Noise_Salt_and_Pepper_2" width="600" height="500">
 

## 2. Image Filtering
We implemented 3 types of Filters: `Average`, `Gaussian`, and `Median` filter. In each filter, you could adjust some parameters such as **mask size** and **Sigma** to show different outputs.

The results below were taken with the following setup:

**Noise parameters:**
- `SNR` = 0.6
- `Sigma` = 128 (For Gaussian Noise Only)

**Filter Parameters:**
- `Mask Size`: 5x5
- `Sigma` = 128 (For Gaussian Filter Only)

The whole GUI is displayed to show you the difference between the noise and the filtered image.

### 2.1 Average Filter Applied on Uniform Noise
<img src="resources/results/image_processing/Filter_Average_On_Noise_Uniform_1.png" alt="Filter_Average_On_Noise_Uniform_1" width="600" height="500">

### 2.2 Gaussian Filter Applied on Gaussian Noise
<img src="resources/results/image_processing/Filter_Gaussian_On_Noise_Gaussian_1.png" alt="Filter_Gaussian_On_Noise_Gaussian_1" width="600" height="500">

### 2.3 Median Filter Applied on Salt & Pepper Noise
<img src="resources/results/image_processing/Filter_Median_On_Noise_Salt_And_Pepper_1.png" alt="Filter_Median_On_Noise_Salt_And_Pepper_1" width="600" height="500">

To increase the blurring effect, increase mask size, and this would be the new output with `mask size = 9x9`.

<img src="resources/results/image_processing/Filter_Gaussian_On_Noise_Gaussian_2.png" alt="Filter_Gaussian_On_Noise_Gaussian_2" width="600" height="500">


## 3. Edge Detection
We implemented 4 types of Edge Detection Techniques (Masks): `Prewitt`, `Sobel`, `Roberts` and `Canny`.

### 3.1 Sobel Mask
<img src="resources/results/image_processing/Edges_Sobel_Mask.png" alt="Edges_Sobel_Mask" 
width="600" height="500">

### 3.2 Roberts Mask
<img src="resources/results/image_processing/Edges_Roberts_Mask.png" alt="Edges_Roberts_Mask" 
width="600" height="500">

### 3.3 Prewitt Mask
<img src="resources/results/image_processing/Edges_Prweitt_Mask.png" alt="Edges_Prweitt_Mask" 
width="600" height="500">

### 3.4 Canny Mask
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.

<img src="resources/results/image_processing/Edges_Canny_Mask.png" alt="Edges_Canny_Mask" 
width="600" height="500">


## 4. Image Histogram and Thresholding
We applied `Histogram Equalization and Normalization`, each algorithm is used for specific problem. In addition to `Convert RGB to Gray Image`. We also applied `Local and Global Thresholding` to differentiate between objects in the image and display specific area of interest. .

### 4.1 Histogram Equalization
<img src="resources/results/image_processing/Histogram_Equalization_1.png" alt="Histogram_Equalization_1"
width="600" height="500">

### 4.2 Histogram Normalization
<img src="resources/results/image_processing/Histogram_Normalization_1.png" alt="Histogram_Normalization_1" 
width="600" height="500">

### 4.3 RGB To Gray
<img src="resources/results/image_processing/Histogram_RGB_To_Gray.png" alt="Histogram_RGB_To_Gray" 
width="600" height="500">

### 4.4 Local Thresholding
<img src="resources/results/image_processing/Thresholding_Local_1.png" alt="Thresholding_Local_1" 
width="600" height="500">

### 4.5 Global Thresholding
<img src="resources/results/image_processing/Thresholding_Global_1.png" alt="Thresholding_Global_1" 
width="600" height="500">


## 5. Hybrid Images
Given 2 images, we apply a Low Pass Filter to the 1st image, and a High Pass Filters to the 2nd image, both in Frequency Domain, and mix the two images to see the output.

### 5.1 Low Pass Filter With High Pass Filter
<img src="resources/results/image_processing/Hybrid_Images_1.png" alt="Hybrid_Images_1" 
width="600" height="500">

If you zoomed in the image you would see more details from the dog, if you zoomed out the image you would see more details from the cat.

<div style="page-break-after: always;"></div>


<!-- Task #2 Report -->

# <a name="boundary-detection_h">Boundary Detection</a>
In this section we present 2 algorithms implementations; `Hough Transformation` and Active Contour Model, aka `'Snake Algorithm'`.


## 1. Hough Transformation
The Hough transform is a technique that locates shapes in images. In particular, it has been used to extract lines, circles and ellipses if you can represent that shape in mathematical form.

The results below were taken with the following setup:

### 1.1 Line Detection
The `Votes` number is basically responsible for determining the amount of output lines. More votes means more detected lines, but this doesn't mean that **5** votes should equal **5** lines, it's not working in that way.

<img src="resources/results/boundary_detection/Hough_Lines_1.png" alt="Hough_Lines_1" width="600" height="500">

<img src="resources/results/boundary_detection/Hough_Lines_2.png" alt="Hough_Lines_2" width="600" height="500">


### 1.2 Circle Detection
Here there is an option to choose the range of radius you want to detect, minimum and maximum range.

<img src="resources/results/boundary_detection/Hough_Circles_1.png" alt="Hough_Circles_1" width="600" height="500">

<img src="resources/results/boundary_detection/Hough_Circles_2.png" alt="Hough_Circles_2" width="600" height="500">

In the 2nd image the maximum radius is less than the bigger circle, so it wasn't detected.


<div style="page-break-after: always;"></div>

## 2. Active Contour Model
Active contour is one of the active models in segmentation techniques, which makes use of the energy constraints and forces in the image for separation of region of interest.

Active contour defines a separate boundary or curvature for the regions of target object for segmentation. This implementation is based on `Greedy Algorithm`.

### 2.1 Result of applying Snake Model on a hand image
<img src="resources/results/boundary_detection/Active_Contour_Snake_1.png" alt="Active_Contour_Snake_1" width="600" height="500">

The parameters' values of `alpha`, `beta`, `gamma` and `number of iterations` are selected by trial and error approach.

### This GIF shows the process in a better way
<img src="resources/results/boundary_detection/Active_Contour_Snake_1.gif" alt="Active_Contour_Snake_1" width="600" height="500">

<div style="page-break-after: always;"></div>


### 2.2 Result of applying the algorithm on circles image
<img src="resources/results/boundary_detection/Active_Contour_Snake_2.png" alt="Active_Contour_Snake_2" width="600" height="500">

### This GIF shows the process in a better way
<img src="resources/results/boundary_detection/Active_Contour_Snake_2.gif" alt="Active_Contour_Snake_2" width="600" height="500">

<div style="page-break-after: always;"></div>


<!-- Task #3 Report -->

# <a name="features-detection-and-image-matching_h">Features Detection and Image Matching</a>
In this section we present 3 algorithms implementations; `Feature Extraction Using Harris Operator`, `Scale Invariant Features (SIFT)` and `Feature Matching`.


## 1. Extract The Unique Features In All Images Using Harris Operator

There are mainly 2 parameters in Harris Detector:
- `Threshold`: Value used computing local maxima (Higher threshold means less corners)
- `Sensitivity`: Sensitivity factor to separate corners from edges. (Small values result in detection of sharp corners).

### 1.1 Harris Corners with `0.2` Threshold and `0.01` Sensitivity
<img src="resources/results/feature_matching/Harris_Corners_1.png" alt="Harris_Corners_1" width="600" height="500">

### 1.2 Harris Corners with `0.1` Threshold and `0.02` Sensitivity
<img src="resources/results/feature_matching/Harris_Corners_2.png" alt="Harris_Corners_2" width="600" height="500">

The processing time is barely noticeable, it only took about `0.01 second` to detect all the corners in the first image and `0.02 second` in the second image.

<div style="page-break-after: always;"></div>

## 2. Feature Descriptors Using Scale Invariant Features (SIFT) Algorithm
Applying SIFT Algorithm to generate features descriptors to use them in matching images with different techniques.

It is not necessary to show the output of SIFT algorithm, the final output is shown in the matching step.

## 3. Matching the Image Set Features 
We applied two Matching Algorithms, Sum Of Squared Differences `(SSD)` and Normalized Cross Correlations `(NCC)`.


### 3.1 Feature Matching Using Sum of Squared Differences (SSD)
<img src="resources/results/feature_matching/Feature_Matching_SSD.png" alt="Feature_Matching_SSD" width="600" height="500">

### 3.2 Feature Matching Using Normalized Cross Correlations (NCC)
<img src="resources/results/feature_matching/Feature_Matching_NCC.png" alt="Feature_Matching_NCC" width="600" height="500">

The computations in this algorithm are heavily and extreme, so as you see it took around `1 minute` to finish the whole process on a small image.

#### Note:
In the above results, each SIFT Algorithm applied was running on a separate thread for faster and better experience, and to avoid GUI freezing problem.



<div style="page-break-after: always;"></div>


<!-- Task #4 Report -->

# <a name="image-segmentation_h">Image Segmentation</a>
In this section we present some algorithms implementations for Image Segmentation; `Using Thresholding and Clustering Methods`.

## 1. Segmentation Using Thresholding
We implemented 3 types of Thresholding, Local and Global for each threshold:
- `Optimal Thresholding`
- `Otsu Thresholding`
- `Spectral Thresholding (More than 2 modes)`

There are mainly 1 parameter used in Local Thresholding:
- `X-Regions`: indicates how many regions we want to divide in x-direction
- `Y-Regions`: indicates how many regions we want to divide in y-direction

### 1.1 Optimal Thresholding
#### Using Global Thresholding
<img src="resources/results/segmentation/Thresholding_Optimal_Global_1.png" alt="Thresholding_Optimal_Global_1" width="600" height="500">

<img src="resources/results/segmentation/Thresholding_Optimal_Global_2.png" alt="Thresholding_Optimal_Global_2" width="600" height="500">

#### Using Local Thresholding
<img src="resources/results/segmentation/Thresholding_Optimal_Local_1.png" alt="Thresholding_Optimal_Local_1" width="600" height="500">

<img src="resources/results/segmentation/Thresholding_Optimal_Local_2.png" alt="Thresholding_Optimal_Local_2" width="600" height="500">

### 1.2 Otsu Thresholding
#### Using Global Thresholding
<img src="resources/results/segmentation/Thresholding_Otsu_Global_1.png" alt="Thresholding_Otsu_Global_1" width="600" height="500">

<img src="resources/results/segmentation/Thresholding_Otsu_Global_2.png" alt="Thresholding_Otsu_Global_2" width="600" height="500">

**Note**: There is some noise in the image which affects the output a little.

#### Using Local Thresholding
<img src="resources/results/segmentation/Thresholding_Otsu_Local_1.png" alt="Thresholding_Otsu_Local_1" width="600" height="500">

### 1.3 Spectral Thresholding
#### Using Global Thresholding
<img src="resources/results/segmentation/Thresholding_Spectral_Global_1.png" alt="Thresholding_Spectral_Global_1" width="600" height="500">

<img src="resources/results/segmentation/Thresholding_Spectral_Global_2.png" alt="Thresholding_Spectral_Global_2" width="600" height="500">

#### Using Local Thresholding
<img src="resources/results/segmentation/Thresholding_Spectral_Local_1.png" alt="Thresholding_Spectral_Local_1" width="600" height="500">

## 2. Segmentation Using Clustering
We implemented 4 Clustering methods:
- `K-Means`
- `Region Growing`
- `Agglomerative Clustering`
- `Mean-Shift`

There are mainly 2 parameters in some Clustering methods:
- `Number of Clusters`: to specify how many clusters you need in the output image.
- `Threshold`: to threshold the output image in specif level in some methods.


### 2.1 K-Means with `6` Clusters
<img src="resources/results/segmentation/Clustering_K_Means_1.png" alt="Clustering_K_Means_1" width="600" height="500">

### 2.2 Region Growing with `3` Clusters
<img src="resources/results/segmentation/Clustering_Region_Growing_1.png" alt="Clustering_Region_Growing_1.png" width="600" height="500">


### 2.3 Agglomerative Clustering with `10` Clusters
<img src="resources/results/segmentation/Clustering_Agglomerative_1.png" alt="Clustering_Agglomerative_1.png" width="600" height="500">

### 2.4 Mean-Shift with `30` Threshold
<img src="resources/results/segmentation/Clustering_Mean_Shift_1.png" alt="Clustering_Mean_Shift_1.png" width="600" height="500">

### Mean-Shift with `90` Threshold
<img src="resources/results/segmentation/Clustering_Mean_Shift_2.png" alt="Clustering_Mean_Shift_2.png" width="600" height="500">

The output image is changed whenever you change the threshold. You could choose the desired threshold by trial and error to know what value fits.

<div style="page-break-after: always;"></div>


<!-- Task #5 Report -->

# <a name="face-detection-and-recognition_h">Face Detection and Recognition</a>
In this section we present Face Detection and Recognition implementations; `Using PCA/Eigenfaces Analysis`.

## 1. Face Detection
This is implemented using openCV library, using `CascadeClassifier` which contains OpenCV data used to detect objects.

There are mainly 2 parameter you can adjust in Face Detection:
- `Scale Factor`: Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
- `Minimum Window Size`: size of each moving window that that algorithm to detect objects.


### Face Detection of one person
<img src="resources/results/face_detection_and_recognition/Face_Detection_1.png" alt="Face_Detection_1" width="600" height="500">

<img src="resources/results/face_detection_and_recognition/Face_Detection_3.png" alt="Face_Detection_3" width="600" height="500">

#### Face Detection of our team
<img src="resources/results/face_detection_and_recognition/Face_Detection_2.png" alt="Face_Detection_3" width="600" height="500">


## 2. Face Recognition
This implementation is based on `PCA/Eigenfaces Analysis`, it's implemented from scratch with the help of some useful libraries.

### Quick Description
- First you need to load the training dataset which consists of 40 class (folder), each class represents one person. Each class has 10 images, taken in different positions.
- Create Eigen-faces matrix for the dataset, which will be used later to compare with any new test image.
- Load your test image then start the matching (recognition) process.
- This function runs in a separate QThreads to ensure best quality and prevent GUI from freezing as it may take some seconds to finish. 

### Face Recognition with correct test
<img src="resources/results/face_detection_and_recognition/Face_Recognition_1.png" alt="Face_Recognition_1" width="600" height="500">

<img src="resources/results/face_detection_and_recognition/Face_Recognition_2.png" alt="Face_Recognition_2" width="600" height="500">

The output image `Best Match`, is just a combination of all the class images in the database, just for displaying purposes to make it clear to the user.

#### Face Recognition with wrong test
<img src="resources/results/face_detection_and_recognition/Face_Recognition_3.png" alt="Face_Recognition_3" width="600" height="500">

<div style="page-break-after: always;"></div>


This repository is created by a group of 4 students in Biomedical Engineering Department, Cairo University. :copyright:


| Name                    | Section | B.N Number   |
|-------------------------|---------|--------------|
| Ahmed Salah El-Dein     | 1       |            5 |
| Ahmad Abdelmageed Ahmad | 1       |            8 |
| Ahmad Mahdy Mohammed    | 1       |            9 |
| Abdullah Mohammed Sabry | 2       |            7 |