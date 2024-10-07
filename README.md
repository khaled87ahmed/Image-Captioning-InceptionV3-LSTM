# Image Captioning with InceptionV3 and LSTM

This repository contains the implementation of an **Image Captioning Model** using a combination of **InceptionV3** (for image feature extraction) and an **LSTM** (for generating captions). The model generates textual descriptions of images using the **Flickr8k Dataset**.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training the Model](#training-the-model)

## Overview
Image captioning is the task of generating a textual description of an image. This model combines deep learning techniques for both **Computer Vision** and **Natural Language Processing** to generate meaningful captions from images. It leverages **InceptionV3** for image feature extraction and **LSTM** for generating captions based on the image features.

## Dataset
The model is trained on the **Flickr8k Dataset**, which consists of:
- 8,091 images
- 5 captions per image

You can download the dataset from Kaggle [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).

## Model Architecture
This project uses a **two-part model**:
1. **Feature Extractor (Encoder)**: A pre-trained **InceptionV3** model is used to extract feature vectors from images. These vectors represent the visual information.
2. **Caption Generator (Decoder)**: An **LSTM-based** network generates a caption, conditioned on the image features and previously generated words.

The architecture includes:
- **Batch Normalization**
- **Embedding Layer** for tokenized captions
- **LSTM** layer for sequential caption generation
- **Dense Layers** with **ReLU** and **Softmax** activations for output

## Installation
To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/khaled87ahmed/image-captioning-inceptionv3-lstm.git
cd image-captioning-inceptionv3-lstm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the Flickr8k dataset from Kaggle and extract it into a folder named **Flickr8k**.


## Training the Model

### Image Features
Features are extracted from the InceptionV3 model, pre-trained on ImageNet, and passed to the caption generator.

### Captions
The captions are tokenized using the `Tokenizer` class from Keras, and each caption is mapped to a sequence of word indices.

### Key Hyperparameters
- **Batch Size**: 64 for training, 32 for validation
- **Learning Rate**: Adaptive learning rate scheduling during training
- **Loss Function**: Categorical Cross-Entropy
- **Optimization**: Adam Optimizer with Gradient Clipping
