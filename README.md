# Food Classification with CNN: Restaurant Recommendation System

This project implements a deep learning–based food classification system using Convolutional Neural Networks (CNNs). By training on a dataset of food images (91 categories), the system learns to classify a user’s food preference.

---

## Table of Contents

1. [What can you do](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Directory layout and dataset import](#dataset-preparation)
5. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Results](#inference--recommendation)
11. [Contributing](#contact)

---

## What can you do

- **Visualize** the project set up (libraries import, seeds setting, dataset loading) and the  hyperparameters employed in the project, with access to the latter for easy modification
- **Build and train** the model, monitoring its accuracy (both on the test and train sets) during the entire process.
- **Evaluate** the final result, both through metrics and a practical simulation

---

## Prerequisites

- **Operating System**: Linux (CentOS 7)
- **GPU**: NVIDIA GPU with CUDA Toolkit **12.6** installed
- **Python**: 3.8 or higher



---

## Installation

1. **Create and activate a Virtual Environment**

   ```bash
   python3 -m venv food_cnn_venv
   source food_cnn_venv/bin/activate
   ```

2. **Update pip to the latest version and install non-standard Python libraries**

   ```bash
   pip install --upgrade pip
   pip install numpy
   pip install matplotlib
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   ```

---

## Directory layout and dataset import

   The dataset folders should be placed in the same repository as the other elements of the project
   ```
   NC_assignment_2425/
   └── food_images/
   └── food_images/
   └── assignment_NC2425
   └── README.md
   └── NC-report.pdf
   
   ```


## Usage

Start the Jupyter server in the terminal using:

```bash
jupyter notebook 

```
This command should automatically launch the server on your default browser, otherwise you can connect at the address: http://localhost:8888/tree

### Training

Launch training from the Jupyter notebook only after running the other data dependencies. To ensure reproducibility within a session, re-run all the previous blocks before resuming the training process. Here is a complete list of the selected hyperparameters for consultation:


```bash
random_seed = 13
workers = 8
batch_size = 32
num_epochs = 1
lr = 0.01. # learning rate
momentum = 0.8
```
### Evaluation

The model training is monitored through the continue computation of training set and test set prediction accuracy. Check the log during training to visualize the changes in real time, or run the following block to print the final test accuracy of the current model without training. The notebook also contains a plot that shows a summary of the accuracy and loss variations during the training.

## Expected results

The model registers the following accuracy scores:

| Dataset     | Test Accuracy |
|-------------------|--------------:|
| Training set      |        88.0%  |
| Test set          |        58.5%  |


---

## Contributing

Laith Agbaria, s4036328 

Gabriele Cybaitė, s3811689

Anna Marini, s3888355

Joudia van Rossum, s3884341

---
