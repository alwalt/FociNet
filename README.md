# FociNet

FociNet is a Convolutional Neural Network (CNN) designed to analyze microscopy images of cells stained with specific markers, such as DAPI, and predict the radiation dose and type. This tool aims to support space scientists in assessing radiation exposure and understanding the biological impacts of space radiation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

FociNet is built using TensorFlow and Keras, leveraging the power of deep learning to automate the analysis of fluorescence microscopy images. It predicts:
- Radiation dose (in Gray)
- Type of radiation (e.g., Fe particles, X-rays)
- Hours post-exposure

This project aims to enhance the efficiency and accuracy of radiation exposure assessment, providing valuable insights for long-term space missions.

## Features

- **Automated Image Analysis**: Analyzes microscopy images to extract meaningful features.
- **Radiation Dose Prediction**: Predicts the radiation dose based on cellular responses.
- **Radiation Type Classification**: Classifies the type of radiation exposure.

## Usage

### Jupyter Notebook

1. **Open the Jupyter Notebook**:
    - Ensure you have Jupyter installed. If not, install it using:
      ```bash
      pip install jupyter
      ```
    - Start Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the `FociNet.ipynb` notebook file from the repository.

2. **Run the Notebook**:
    - Follow the instructions in the notebook to load data, preprocess images, train the model, and evaluate the results.


