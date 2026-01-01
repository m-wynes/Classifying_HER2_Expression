# Classifying_HER2_Expression

## Overview
This project uses a convolutional neural network (CNN) to classify HER2 expression levels in H&E-stained breast cancer biopsy slides.

## Dataset
The dataset was obtained from the Breast Cancer Immunohistochemistry (BCI) project, and contains H&E-stained histopathology images labeled for HER2 expression: https://bupt-ai-cz.github.io/BCI/

## File Descriptions
### data_preprocessing.py
This file handles dataset preparation. Classes are defined based on image filenames, data is split into training and testing sets, and PyTorch data loaders are created for batch processing.

### train_and_eval_model.ipynb
This file visualizes sample training images, defines the CNN architecture, and handles model training and performance evaluation. Gradient-weighted Class Activation Mapping ([Grad-CAM](https://arxiv.org/abs/1610.02391)), an explainability technique, has been added to visualize which image regions influence the model’s classification.

### train_and_eval_model_output.ipynb
This file displays the outputs from the train_and_eval_model.ipynb

## Results and Discussion
The training set’s accuracy metric indicates that 91.5761% of predictions were correct. With a macro precision of 87.8108%, the model is reliable in ensuring the samples it predicts for each class truly belong to that class. Finally with a macro recall of 93.3633%, the model is largely able to correctly identify most samples from each class, suggesting strong sensitivity for all four classes.

The validation set’s accuracy metric indicates that 88.0218% of predictions were correct, suggesting the model generalizes well with unseen data. The macro precision of 82.2034% indicates that the samples the model predicts for each class mostly belong to that class. Finally, with a macro recall of 90.4287%, the model is largely able to correctly identify most samples from each class, suggesting strong sensitivity for all four classes in unseen data.

Additionally, the Multiclass ROC Curve figures show that each class achieves an AUC score ≥ 0.95 in both the training and validation sets. This performance metric further indicates that the model performs well and can adequately predict HER2 expression levels.


