# Skin Cancer Classification using Deep Learning

A deep learning project for classifying skin lesions into three categories: Basal Cell Carcinoma (BCC), Melanoma, and Nevus using transfer learning with MobileNetV2.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture]()
- [Results]()
- [Project Structure]()
- [Requirements]()
- [Performance Metrics]()

## 🔍 Overview

This project implements a Convolutional Neural Network (CNN) based on MobileNetV2 architecture to classify skin lesions into three categories:
- **BCC (Basal Cell Carcinoma)** - A type of skin cancer
- **Melanoma** - A serious form of skin cancer
- **Nevus** - Non-cancerous moles

The model achieves high accuracy in distinguishing between cancerous and non-cancerous skin lesions, making it a potential tool for assisting dermatologists in early diagnosis.

## 📊 Dataset

The dataset is organized into three main categories with separate train and test splits:

```bash

data/
├── basal-cell-carcinoma/
│   ├── train/
│   └── test/
├── melanoma/
│   ├── train/
│   └── test/
└── nevus/
    ├── train/
    └── test/
```
I used [ISIC 2019 Dataset](https://challenge.isic-archive.com/data/#2019) and all images are preprocessed and resized to 224x224 pixels for model input.

## 🏗️ Model Architecture

The model uses MobileNetV2 as the base architecture with custom classification layers:

```bash
MobileNetV2 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.1)
    ↓
Dense(3, activation='softmax')
```

### Key Features:

- **Transfer Learning**: Utilizes MobileNetV2 pretrained weights
- **Fine-tuning**: All layers are trainable for optimal performance
- **Data Augmentation**: Includes rotation, zoom, and shift transformations
- **Regularization**: Dropout layers to prevent overfitting
- **Callbacks**: ModelCheckpoint and ReduceLROnPlateau for optimal training


## 📈 Results
### Training Performance

The model demonstrates excellent performance across all classes:

- **BCC**: AUC = 0.99
- **Melanoma**: AUC = 0.95
- **Nevus**: AUC = 0.97
- **Micro-average**: AUC = 0.98
- **Macro-average**: AUC = 0.97

### Validation Results

Normalized Performance:

- **BCC**: 94% accuracy (624/665 correct predictions)
- **Melanoma**: 78% accuracy (705/904 correct predictions)
- **Nevus**: 92% accuracy (2379/2575 correct predictions)

### Test Results

Test Set Performance:

- **BCC**: 96% accuracy (48/50 correct predictions)
- **Melanoma**: 90% accuracy (45/50 correct predictions)
- **Nevus**: 96% accuracy (48/50 correct predictions)

### Classification Report

```bash
              precision    recall  f1-score   support

         BCC     0.96      0.96      0.96        50
    Melanoma     0.94      0.90      0.92        50
       Nevus     0.89      0.96      0.92        50

    accuracy                         0.94       150
   macro avg     0.93      0.94      0.93       150
weighted avg     0.93      0.94      0.93       150

```

## 📁 Project Structure

```bash
skin-cancer-classification-mobilenetv2/
│
├── data/                          # Dataset directory
│   ├── basal-cell-carcinoma/
│   ├── melanoma/
│   └── nevus/
│
├── features-train.npy             # Preprocessed training features
├── labels-train.npy               # Training labels
├── features-test.npy              # Preprocessed test features
├── labels-test.npy                # Test labels
│
├── model_v1.h5.keras              # Trained model
├── .mdl_wts.hdf5.keras           # Best model checkpoint
│
├── train.py                       # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```
## 📊 Performance Metrics

### Overall Accuracy
- **Validation Set**: ~88% accuracy
- **Test Set**: ~94% accuracy


### Per-Class Performance (Test Set)

Class | Precision | Recall | F1-Score
--- | --- | --- | ---
BCC |	0.96 | 0.96 |	0.96
Melanoma | 0.94 |	0.90 | 0.92
Nevus | 0.89 | 0.96 | 0.92


### Key Observations

- Excellent performance in detecting BCC (96% accuracy)
- High sensitivity for Nevus detection (96% recall)
- Melanoma classification shows good balance between precision and recall
- Very low false positive rate across all classes



## 🎯 Model Training Details

### Hyperparameters


- Input Size: 224x224x3
- Batch Size: 64 (training), 16 (evaluation)
- Epochs: 30 (with early stopping)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Learning Rate: Adaptive (ReduceLROnPlateau)

### Data Augmentation


- Rotation: ±10 degrees
- Zoom: ±10%
- Width/Height Shift: ±10%
- No horizontal/vertical flipping



## 🔬 Future Improvements


- [ ] Implement ensemble methods with multiple architectures
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Expand dataset with additional skin lesion types
- [ ] Deploy as web application or mobile app
- [ ] Add confidence thresholds for uncertain predictions
- [ ] Implement cross-validation for more robust evaluation


## 🙏 Acknowledgments

- MobileNetV2 architecture by Google
- ImageNet pretrained weights
- Skin lesion dataset contributors














