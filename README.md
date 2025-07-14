# Water Quality Classification with PyTorch

This project demonstrates a complete workflow for a multi-class classification task using PyTorch. The goal is to predict the quality of water based on various chemical and physical properties. It showcases essential PyTorch functionalities, from data preparation to model training, evaluation, and deployment.

---

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [PyTorch Implementation](#-pytorch-implementation)
  - [Custom Dataset Class](#1-custom-dataset-class)
  - [Neural Network Architecture](#2-neural-network-architecture)
  - [Training & Evaluation Loop](#3-training--evaluation-loop)
  - [Model Persistence](#4-model-persistence)
  - [Inference](#5-inference)
- [How to Use](#-how-to-use)

---

## 📊 Dataset

The project uses the `water_quality.csv` dataset, which contains various features to classify water quality into 5 categories: `Excellent`, `Good`, `Poor`, `Very Poor yet Drinkable`, and `Unfit for Drinking`.

**Features include:** `State`, `Year`, `pH`, `EC`, `TH`, `WQI`, and concentrations of various minerals like `Ca`, `Mg`, `Na`, etc.

---

## ⚙️ Data Preprocessing

Before training, the data is preprocessed using `scikit-learn`. A `ColumnTransformer` is used to apply different transformations to different types of columns:
* **Numerical Features**: Scaled using `StandardScaler`.
* **Categorical Features**: Encoded using `OneHotEncoder`.

This `preprocessor` is a critical component and is saved alongside the model. It ensures that any new data for prediction is transformed in the exact same way as the training data.

---

## 🔥 PyTorch Implementation

This project highlights several core PyTorch functionalities.

### 1. Custom Dataset Class
A `CustomDataset` class is created by inheriting from `torch.utils.data.Dataset`. This is the standard PyTorch way to wrap a dataset. It implements three essential methods:
* `__init__()`: Initializes the dataset and stores the feature and label tensors.
* `__len__()`: Returns the total number of samples in the dataset.
* `__getitem__()`: Retrieves a single sample (feature and label pair) at a given index.

This class allows the `DataLoader` to efficiently create shuffled mini-batches for training.

```python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```
### 2. Neural Network Architecture
A flexible neural network is defined using `torch.nn.Module`. This project demonstrates:
* **Multiple Linear Layers**: A deep network with multiple hidden layers (`nn.Linear`).
* **Activation Functions**: Use of different activation functions like `F.relu` and `F.tanh` in different layers.
* **Dropout for Regularization**: Implementation of `nn.Dropout` with varying probabilities after hidden layers to prevent overfitting.

```python
class MultiClassClassifier(nn.Module):
    def __init__(self, input_features):
        super(MultiClassClassifier, self).__init__()
        self.layer1 = nn.Linear(input_features, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 5) # 5 output classes

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.tanh(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
```

### 3. Training & Evaluation Loop
A comprehensive function, `train_and_evaluate_model`, encapsulates the entire training process. This function showcases:
* **Batch Iteration**: Using `DataLoader` to iterate over the data in batches.
* **Optimizer**: Use of the `Adam` optimizer with `weight_decay` (L2 regularization) to update model parameters.
* **Loss Function**: `nn.CrossEntropyLoss` is used, which is standard for multi-class classification and expects raw logits from the model.
* **Gradient Management**: The classic training steps: `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
* **Switching Modes**: Correctly using `model.train()` and `model.eval()` to ensure layers like Dropout behave as expected during training and are disabled during evaluation.
* **Live Metrics**: Calculation and printing of both training and validation accuracy and loss for each epoch.

### 4. Model Persistence
The workflow demonstrates how to properly save and load model components for future use:
* **Saving the Model**: The trained model's parameters (`state_dict`) are saved using `torch.save()`.
* **Saving the Preprocessor**: The fitted `scikit-learn` `preprocessor` is saved using `pickle`. This is crucial because the model and the preprocessor are a pair.

### 5. Inference
The final part of the project shows how to use the saved artifacts to make a prediction on new, raw data:
1.  Load the saved `preprocessor` and `model`.
2.  Place the raw new data into a pandas DataFrame.
3.  Use the **loaded preprocessor** to transform the raw data.
4.  Convert the transformed data into a PyTorch tensor.
5.  Feed the tensor into the **loaded model** to get a prediction.

---

## 🚀 How to Use

1.  **Train the Model**: Run the `PyTorch_Project.ipynb` notebook from top to bottom. This will train the model, save `water_quality_classifier.pth`, and save `preprocessor.pkl`.
2.  **Make Predictions**: Use the code snippets in the "Inference" section of the notebook to load the saved files and predict new water quality samples.
