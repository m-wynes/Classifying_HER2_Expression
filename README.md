# Classifying_HER2_Expression

## Overview
This project uses a convolutional neural network (CNN) to classify HER2 expression levels in H&E-stained breast cancer biopsy slides.

## Dataset
The dataset was obtained from the Breast Cancer Immunohistochemistry (BCI) project, and contains H&E-stained histopathology images labeled for HER2 expression: https://bupt-ai-cz.github.io/BCI/

## File Descriptions
### data_preprocessing.py
This file handles dataset preparation. Classes are defined based on image filenames, data is split into training and testing sets, and PyTorch data loaders are created for batch processing.

### train_and_eval_model.ipynb
This file visualizes sample training images, defines the CNN architecture, and handles model training and performance evaluation.

### train_and_eval_model_outputs.ipynb
This file displays the outputs from the train_and_eval_model.ipynb
