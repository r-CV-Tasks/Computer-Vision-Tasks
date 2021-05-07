<h1 style="text-align: center;"> Features Detection and Image Matching</h1>
<h3 style="text-align: center;"> Submitted to: Dr. Ahmed Badawi</h3>
<h3 style="text-align: center;"> 2020 - 2021</h3>

| Name                    | Section | B.N Number   |
|-------------------------|---------|--------------|
| Ahmed Salah El-Dein     | 1       |            5 |
| Ahmad Abdelmageed Ahmad | 1       |            8 |
| Ahmad Mahdy Mohammed    | 1       |            9 |
| Abdullah Mohammed Sabry | 2       |            7 |


## Table of content
##### 1. Extract the unique features in all images using Harris operator
##### 2. Generate feature descriptors using scale invariant features (SIFT)
##### 3. Match the image set features using sum of squared differences (SSD) and normalized cross correlations

<div style="page-break-after: always;"></div>

**Each Algorithm Applied was thrown onto a thread for faster better experience**
### Extract the unique features in all images using Harris operator
**Applying a threshold 0.2 it only took about 0.01 second to detect all Image Corners**
![harris_v1](./resources/results/task%203/harris_v1.png)

**We also applied the harris operator on a harder image with the same threshold this time it only took 0.02 seconds** 
![harris_v2](./resources/results/task%203/harris_v2.png)


[comment]: <> (<img src="resources/results/task2/canny_edges_result2.png" alt="Canny Edges Result2" width="700" height="400">)
<div style="page-break-after: always;"></div>

### Using Sift Descriptors and Harris Operator to Match the image set features 
**Using Harris Operator to Detect Image Key Points and Applying the SIFT Algorithm to Generate each Feature Descriptor,
Applying Two Matching Algorithms SSD And NCC**
### using sum of squared differences (SSD)
![ssd](./resources/results/task%203/ssd.png)
### Using Normalized Cross Correlations (NCC)
![ssd](./resources/results/task%203/ncc.png)


<div style="page-break-after: always;"></div>
