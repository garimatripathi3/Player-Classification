# Player Image Classification  Using Deep Features and Color Histograms

## Overview
Code  clusters images of players based on their clothing using a combination of deep learning features from MobileNetV2 and color histograms of specific body regions (upper and lower body). 

## Table of Contents
- [Overview](#overview)
- [Workflow Summary](#workflow-summary)
- [Execution Time](#execution-time)
- [Important Notes](#important-notes)

## Workflow Summary

1. **Deep Learning Feature Extraction**
    Utilizes **MobileNetV2**, a pre-trained  lightweight deep learning model, to extract high-level features from each image. 

2. **Color Histogram Extraction**
    Color histograms are extracted from the **upper body** and **lower body** regions of the image to capture the color distribution in specific body areas.

3. **Similarity Calculation**
    The histograms of the upper and lower body regions are compared using **histogram correlation** to determine which region has more distinct visual information.

4. **Flagging Process**
    A **flagging system** counts how many images have more distinct upper body features versus lower body features. The region with the highest count is selected for final feature extraction across the dataset.

5. **Feature Combination**
    The selected histogram (upper or lower body) is combined with the deep features extracted by MobileNetV2. This combined feature vector represents the unique characteristics of each image.

6. **Agglomerative Clustering**
    The combined feature vectors are used to cluster the images into **two groups** using **Agglomerative Clustering**.


## Execution Time

Clustering took 0.0201 seconds
Total time for the entire process including saving of images: 6.7 second

## Important Notes
**MobileNetV2**: This model was chosen for its lightweight architecture, making it faster and more efficient compared to heavier models like ResNet. 
  
 **Optimized Color Histogram Extraction**: The number of histogram bins has been reduced to 4, which helps speed up the process.

 **Upper and Lower Body Analysis**: Both the upper and lower body regions are analyzed separately to provide richer feature extraction, allowing for more accurate clustering based on clothing differences.

**Optimized Saving Process**: The saving procedure for clustered images has been optimized using parallel processing with `ThreadPoolExecutor`, reducing the overall time for saving images, especially when handling larger datasets.


