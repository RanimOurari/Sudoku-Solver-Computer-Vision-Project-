# Sudoku Solver Computer Vision Project
This project was developed as part of the Computer Vision course in the Joint Master of Computer Science program in Switzerland. It aims to solve a Sudoku puzzle directly from an input image, implementing a full computer vision pipeline to detect, extract, and recognize handwritten digits before sending the puzzle to a provided solver.

## Project Overview

The goal of the assignment was to build an end-to-end system capable of:

* **Preprocessing a Sudoku image** (grayscale conversion, thresholding, contour detection, and frontalization)
* **Extracting and segmenting the 81 grid cells**
* **Recognizing digits** using a template-matching approach
* **Reconstructing the Sudoku board** from the recognized digits
* **Solving the puzzle** using a provided solver algorithm

Only the image processing and digit recognition pipelines were implemented by me; the Sudoku-solving logic was supplied by the instructor.

## Technologies & Libraries

* **Python**
* **OpenCV:** image preprocessing, contour detection
* **scikit-image:** thresholding, filters
* **NumPy:** numerical operations
* **matplotlib:** visualization and debugging

## Implementation Structure

* `frontalization.py` grid detection, perspective correction, preprocessing
* `recognition.py` digit extraction, template matching, digit classification
* `template.py` creation and management of digit templates

## Features

* Automatic grid detection and frontalization
* Noise-resistant preprocessing pipeline
* Template-based digit recognition (up to 2 templates per digit)
* Full integration with the provided Sudoku solver
* Test and template generation scripts for reproducibility

## Evaluation Workflow

1. Run `create_templates.py` to generate digit templates
2. Run `test.py` to validate the full solution pipeline

## Challenges

* **Robust grid detection:** Handling images with skew, perspective distortion, or shadows required careful contour filtering and reliable corner extraction.
* **Digit segmentation:** Ensuring consistent extraction of digits across varying handwriting styles and image conditions.
* **Template matching limitations:** Creating representative digit templates without exceeding the “two per digit” constraint required tuning and trial-and-error.
* **Noise and artifacts:** Thresholding and denoising had to be precise enough to avoid misclassification while preserving digit shapes.

## Learnings

* Gained hands-on experience in **practical computer vision pipelines**, from preprocessing to recognition.
* Learned how to use **OpenCV transformations** (e.g., adaptive thresholding, contour detection, perspective warping).
* Better understanding of **template matching** and its constraints compared to modern ML-based recognition.
* Developed debugging strategies for image-based pipelines using visualization tools.
* Improved ability to write clean, modular, and maintainable code within strict assignment constraints.



