# AI Art vs. Human Art Classification

This project aims to build a Convolutional Neural Network (CNN) model to classify images as either AI-generated art or human-created art.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Project Constants](#project-constants)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Prediction](#prediction)

## Project Overview

The goal of this project is to develop a deep learning model capable of distinguishing between artworks created by artificial intelligence and those created by human artists. This is achieved by training a CNN on a dataset comprising both types of art.

## Dataset

The dataset used for this project is "AI Art vs Human Art" from Kaggle. It contains images categorized into `AiArtData` and `RealArt` (Human Art). The dataset is downloaded and extracted directly within the notebook.

- **Source**: [Kaggle: AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art)
- **Classes**: `AiArtData` (AI-generated art), `RealArt` (Human art)
- **Total Images**: 971

## Setup and Installation

To run this notebook, you'll need a Google Colab environment or a local Python environment with the following libraries:

1.  **Clone the Repository (if applicable) or Open in Colab.**
2.  **Install Dependencies**:

    ```bash
    !pip install opendatasets --quiet
    ```

3.  **Kaggle Credentials**: The dataset is downloaded from Kaggle. You will be prompted to enter your Kaggle username and API key. Follow the instructions provided by `opendatasets`.

    ```python
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art")
    ```

4.  **Unzip the dataset**:

    ```python
    from zipfile import ZipFile
    with ZipFile("ai-art-vs-human-art.zip", "r") as file:
      file.extractall("data")
    ```

## Project Constants

The following constants are defined for consistency throughout the project:

```python
RAW_DIR="data/Art"
IMG_HEIGHT=180
IMG_WIDTH=180
EPOCH=10
BATCH_SIZE=16
SEED=42
CHANNELS=3
```

## Model Architecture

The model is a Sequential CNN designed for image classification. It includes several convolutional layers, batch normalization, max-pooling layers, dropout for regularization, and dense layers for classification. The input images are resized to `180x180` pixels with 3 channels.

```python
model=Sequential([
    layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH, CHANNELS)),
    layers.Rescaling(1./255.),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Changed to 1 for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['BinaryAccuracy'])
```

## Training and Evaluation

The model was trained for `EPOCH=10` epochs with a `BATCH_SIZE=16`. The dataset was split into training, validation, and test sets using an 80/10/10 split.

-   **Training Accuracy (BinaryAccuracy)**: The final training accuracy achieved after 10 epochs.
-   **Validation Accuracy (val_BinaryAccuracy)**: The final validation accuracy achieved after 10 epochs.
-   **Loss**: Binary Crossentropy.

Training logs and plots for accuracy and loss are generated to visualize the model's performance over epochs.

## Prediction

A prediction pipeline is provided to load and preprocess an image, then use the trained model to predict its class.

```python
def load_preprocess(image, resize_shape=(180,180)):
  img=Image.open(image)
  resize_img=img.resize(resize_shape)
  np_img=np.array(resize_img)
  batch_img=np.expand_dims(np_img, axis=0)
  norm_img=batch_img.astype('float32')/255.
  return norm_img

def prediction(model, image, class_name):
  preprocessed_img=load_preprocess(image)
  plt.imshow(np.array(preprocessed_img[0]))
  plt.axis('off')
  pred_image=model.predict(preprocessed_img)
  pred_index=np.argmax(pred_image, axis=1)[0]
  pred_img_class=class_name[pred_index]
  return pred_img_class

# Example Usage:
class_indx={0: 'AiArtData', 1: 'RealArt'}
img_path="/content/aiart.jpeg" # Replace with your image path
prediction(model, img_path, class_indx)
```

