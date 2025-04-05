# Fashion-Mnist-Classifier

A simple yet effective deep learning project built with Keras to classify fashion items. It demonstrates the complete workflow of an image classification task using a fully connected neural network, from data preprocessing to model evaluation and prediction.

## Overview

This project covers:

- Loading and inspecting the Fashion MNIST dataset
- Preprocessing data (normalization and one-hot encoding)
- Visualizing sample images
- Building and training a neural network using Keras
- Evaluating the model on unseen test data
- Making predictions and visualizing results

## How It Works

1. Data Loading and Visualization
&nbsp;&nbsp;&nbsp;&nbsp;- We load the dataset using keras.datasets.fashion_mnist, print its shape, and visualize a sample image to understand the structure.

3. Data Preprocessing
  &nbsp;&nbsp;&nbsp;&nbsp;- Normalize pixel values to the range [0, 1] to help the model converge faster.
  &nbsp;&nbsp;&nbsp;&nbsp;- Convert labels to one-hot encoded vectors using to_categorical for classification.

3. Model Architecture
  - We use a fully connected neural network

4. Training
  The model is compiled using:

  - Loss: CategoricalCrossentropy

  - Optimizer: Adam

  - Metric: CategoricalAccuracy

  * Training is performed over 50 epochs with a batch size of 50 and 20% of training data used for validation. *

5. Evaluation and Prediction
  - We evaluate the model on the test set, make predictions, and visualize the results with the predicted and true labels.

## Requirements
  pip install -r requirements.txt
