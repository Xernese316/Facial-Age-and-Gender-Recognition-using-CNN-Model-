# Age and Gender Prediction using CNN

This project aims to build a Convolutional Neural Network (CNN) model for predicting age and gender from facial images. The model is trained on the UTKFace dataset, which contains over 23,000 images of faces with age and gender labels.

## Dataset

The UTKFace dataset is loaded and preprocessed, with the image paths, age labels, and gender labels stored in a pandas DataFrame. The gender labels are mapped to 'Male' and 'Female' for better readability.

## Exploratory Data Analysis

The project includes an exploratory data analysis step, where various visualizations are created to understand the dataset better. These include:

- Displaying a sample image from the dataset
- Plotting the distribution of ages
- Showing a grid of sample images with their corresponding age and gender labels

## Feature Extraction

A custom function `extract_features` is defined to extract features from the images. It resizes the images to 128x128 pixels and converts them to grayscale. The extracted features are then reshaped and normalized before being used as input to the CNN model.

## Model Architecture

The CNN model architecture is defined using the Keras functional API. The model consists of the following layers:

- Convolutional layers with increasing filters (32, 64, 128, 256)
- Max Pooling layers for downsampling
- Flatten layer to convert the feature maps to a 1D vector
- Dense layers (256 units) with dropout for regularization
- Two output layers: one for gender prediction (binary classification) and one for age prediction (regression)

The model is compiled with a binary cross-entropy loss for gender prediction, mean absolute error (MAE) loss for age prediction, and the Adam optimizer. The metrics used are accuracy, MAE, and gender accuracy.

## Model Training

The model is trained on the extracted features and corresponding labels for 30 epochs, with a validation split of 0.2. The training history is stored for further analysis and visualization.

## Usage

To run this project, you'll need to have the UTKFace dataset downloaded and the file paths updated accordingly. Additionally, the required Python libraries (listed at the beginning of the code) need to be installed.

Once set up, you can run the code to train the model and evaluate its performance on the validation set.

Note: The provided code is a snapshot and may require additional modifications or improvements depending on your specific requirements.