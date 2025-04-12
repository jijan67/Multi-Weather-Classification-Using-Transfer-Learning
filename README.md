# 🌦️ Multi-Weather Classification Using Transfer Learning

This documentation provides a detailed explanation of a Multi-Weather Classification system implemented using VGG19 transfer learning and logistic regression in Python. The goal is to accurately classify weather conditions from images across 7 different classes.

---

## Overview

The project uses the MultiWeatherDataset containing images of different weather conditions (Cloudy, Fog, Rain, Sand, Shine, Snow, and Sunrise). The model employs VGG19 for feature extraction and logistic regression for classification, achieving an overall accuracy of 92%.

## Libraries Used

The following Python libraries were used to implement the model:

- **NumPy:** For numerical operations and array handling
- **Pandas:** For data manipulation and analysis
- **Matplotlib:** For data visualization
- **Scikit-learn:** For machine learning algorithms and evaluation metrics
- **TensorFlow/Keras:** For loading and utilizing the pre-trained VGG19 model

## Project Structure

**1. Data Loading and Organization:**
- The dataset is loaded from the MultiWeatherDataset directory
- File paths and corresponding labels are collected and organized into a DataFrame
- Data is split into training and test sets using stratified sampling

**2. Feature Extraction:**
- A pre-trained VGG19 model is used to extract high-level features from the images
- Images are preprocessed to meet VGG19 input requirements (224×224 pixels)
- Features are flattened for use with logistic regression

**3. Model Training:**
- Features are standardized using StandardScaler
- A multinomial logistic regression model is trained on the extracted features
- The model is configured with 'lbfgs' solver and 1000 maximum iterations

**4. Model Evaluation:**
- Accuracy and loss metrics are calculated for both training and test sets
- Detailed classification reports showing precision, recall, and F1-score
- Confusion matrix visualization to understand misclassifications
- ROC curves and AUC scores for each class

## Evaluation Results

- **Test Accuracy:** 92%
- **Class Performance (F1-scores):**
  - Cloudy: 0.93
  - Fog: 0.83
  - Rain: 0.99
  - Sand: 0.88
  - Shine: 0.93
  - Snow: 0.93
  - Sunrise: 0.96

The model demonstrates strong performance across most classes, with Rain showing the highest F1-score (0.99) and Fog showing the lowest (0.83).

## Implementation Details

The project utilizes transfer learning, leveraging the feature extraction capabilities of VGG19, which was pre-trained on ImageNet. This approach allows the model to benefit from patterns already learned from millions of images, significantly enhancing classification performance without needing to train a deep neural network from scratch.

The extracted features are then fed into a multinomial logistic regression classifier, which provides a good balance between computational efficiency and classification performance for this multi-class problem.

## Visualizations

The project includes:
- Confusion matrix to identify patterns of misclassification
- ROC curves for each class to evaluate binary classification performance
- Area Under Curve (AUC) calculations to quantify discriminative power

## Conclusion

The multi-weather classification system demonstrates high accuracy (92%) in classifying weather conditions from images. The combination of VGG19 feature extraction and logistic regression proves to be an effective approach for this image classification task, providing reliable predictions across diverse weather conditions.

---
