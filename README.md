# Impact of Synthetic Data on performance of Image Classifiers

This project uses GANs to generate synthetic image data and evaluates its impact on CNN image classification. By comparing a CNN trained on real data alone to one trained with both real and GAN-generated data, it assesses the effectiveness of GAN-based augmentation using metrics like accuracy, F1-score, and confusion matrices.

# Setup

Intialize and activate a Python Virtual Environment

```
python -m venv venv
source venv/bin/activate
```

Install Dependencies

```
pip install -r requirements.txt
```

# Directory Structure

### The resulting directory structure

The directory structure of your new project will look something like this (depending on the settings that you choose):

```
├── data               <- data from 3rd party resources
│
├── models             <- Trained model params
│
├── results            <- Generated images/pdfs of training loss curve and confusion matrices
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
└── src   <- Source code for use in this project.
    │
    ├── config.py                 <- Stores useful constant variables
    │
    ├── main.py                   <- Entry point to run entire experiment
    │
    ├── cnn                       <- Directory for the CNN classifier
    |   |
    |   ├── config.py             <- Stores training hyperparams
    |   |
    |   ├── models.py             <- Define the model architectures for the CNN
    |   |
    |   ├── train.py              <- Train and evaluate the CNN
    |   |
    |   ├── dataset_dispatcher.py <- Mapping mechanism used to map a dataset to correct handler method
    |   |
    |   └── data_handler.py       <- Data preprocessing methods
    |
    └──gan                        <- Directory for the GAN
        |
        ├── datasets.py           <- Scripts for handling dataset preprocessing logic
        |
        ├── models.py             <- Define the model architectures for the GAN
        |
        ├── train.py              <- Train the GAN
        |
        └── utils.py              <- Collection of important helper functions
```
