# Fashion-MNIST Classification using PyTorch ANN

A simple Artificial Neural Network (ANN) implemented with **PyTorch** for classifying Fashion-MNIST images into 10 different clothing categories. This project demonstrates the complete deep learning workflow, including data preprocessing, custom dataset creation, mini-batch training, forward propagation, loss calculation, backpropagation, optimization, and model evaluation.

## 📌 Project Overview

The model takes a **28 × 28 grayscale Fashion-MNIST image** as input and classifies it into one of 10 categories.

Since each image contains:

$$
28 \times 28 = 784
$$

pixels, each image is represented as a **784-dimensional feature vector**.

The ANN architecture is:

```text
784 Input Features
        ↓
Fully Connected Layer (128 neurons)
        ↓
ReLU
        ↓
Fully Connected Layer (64 neurons)
        ↓
ReLU
        ↓
Fully Connected Layer (10 neurons)
        ↓
10 Class Scores
```

## 🧠 Model Architecture

The network is implemented using PyTorch's `nn.Sequential`:

```python
self.model = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

### Layer Details

| Layer           | Input | Output | Activation |
| --------------- | ----: | -----: | ---------- |
| Input           |   784 |    784 | —          |
| Fully Connected |   784 |    128 | ReLU       |
| Fully Connected |   128 |     64 | ReLU       |
| Output          |    64 |     10 | —          |

The final layer produces **10 logits**, one for each Fashion-MNIST class.

## 👕 Fashion-MNIST Classes

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

## ⚙️ Data Processing

The dataset is loaded from a CSV file:

```python
df = pd.read_csv('/content/fmnist_small.csv')
```

The first column contains the class labels, while the remaining 784 columns contain pixel values.

```python
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
```

The dataset is divided into training and testing sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Pixel values are normalized from the range **0–255** to **0–1**:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## 📦 Custom Dataset and DataLoader

A custom PyTorch `Dataset` is used to convert the NumPy arrays into PyTorch tensors.

```python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

The `DataLoader` divides the training data into mini-batches of 32 samples:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)
```

The testing data is loaded without shuffling:

```python
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)
```

## 🔄 Training Process

The model is trained for **10 epochs** using **Stochastic Gradient Descent (SGD)**.

```python
epochs = 10
learning_rate = 0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

For every batch, the training process follows:

```text
Input Batch
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Clear Previous Gradients
    ↓
Backpropagation
    ↓
Update Weights
```

The core training steps are:

```python
outputs = model(batch_features)
loss = criterion(outputs, batch_labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### What happens during training?

**1. Forward Pass**

The input passes through all layers and produces 10 output scores.

**2. Loss Calculation**

`CrossEntropyLoss` compares the predicted scores with the actual class labels.

**3. Backpropagation**

```python
loss.backward()
```

calculates the gradients of the loss with respect to the model parameters.

**4. Weight Update**

```python
optimizer.step()
```

updates the weights using the calculated gradients.

The basic update concept is:

$$
W_{\text{new}}
=
W_{\text{old}}
-
\eta
\frac{\partial L}{\partial W}
$$

where:

* \(W\) = model weight
* \(\eta\) = learning rate
* \(L\) = loss

This process is repeated for every batch and every epoch so that the model gradually learns useful patterns from the training data.

## 📊 Evaluation

After training, the model is switched to evaluation mode:

```python
model.eval()
```

Gradient calculation is disabled during evaluation:

```python
with torch.no_grad():
```

For each sample, the class with the highest output logit is selected:

```python
_, predicted = torch.max(outputs, 1)
```

The model is evaluated on both:

* Training dataset
* Testing dataset

This provides training and test accuracy for evaluating the model's performance.

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## 📁 Project Structure

```text
Fashion-MNIST-ANN/
│
├── fmnist_small.csv
├── fashion_mnist_ann.py
└── README.md
```

> The dataset file name and Python script name can be modified according to the actual files in the repository.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AnikHasan2356/<repository-name>.git
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

### 3. Place the dataset

Make sure `fmnist_small.csv` is available in the expected project directory.

If necessary, update:

```python
df = pd.read_csv('/content/fmnist_small.csv')
```

with the correct dataset path.

### 4. Run the Python script

```bash
python fashion_mnist_ann.py
```

## 🎯 Learning Objectives

This project was developed to understand the fundamental workflow of Artificial Neural Networks using PyTorch, including:

* Loading and preprocessing image datasets
* Train-test splitting
* Feature normalization
* Creating custom PyTorch datasets
* Using `DataLoader` and mini-batch training
* Building an ANN with fully connected layers
* Understanding forward propagation
* Calculating classification loss
* Backpropagation
* Gradient-based weight optimization
* Epochs and batch-wise training
* Model evaluation
* Training vs. testing accuracy

## 📌 Key Concept

The complete learning process can be summarized as:

```text
Dataset
   ↓
Train/Test Split
   ↓
Normalization
   ↓
Custom Dataset
   ↓
DataLoader
   ↓
ANN
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Weight Update
   ↓
Repeat for Multiple Epochs
   ↓
Evaluation
```

## 📄 License

This project is intended for educational and learning purposes.
