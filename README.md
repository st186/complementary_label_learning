# Complementary-Label Learning

This repository gives the implementation for *complementary-label learning* from the ICML 2019 paper [1], the ECCV 2018 paper [2], and the NeurIPS 2017 paper [3].

## Requirements
- Python 3.6
- numpy 1.14
- PyTorch 1.1
- torchvision 0.2
- Scikit-Learn 0.22

## Demo
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```bash

## Dataset - https://www.kaggle.com/c/prudential-life-insurance-assessment/data

Download the dataset and put it inside data folder

## Install all the requirements -

pip3 install -r requirements.txt

python3 demo.py --method forward --model resnet
```
#### Methods and models
In `demo.py`, specify the `method` argument to choose one of the 5 methods available:

- `ga`: Gradient ascent version (Algorithm 1) in [1].
- `nn`: Non-negative risk estimator with the max operator in [1].
- `free`: Assumption-free risk estimator based on Theorem 1 in [1].
- `forward`: Forward correction method in [2].
- `pc`: Pairwise comparison with sigmoid loss in [3].

Specify the `model` argument:

- `linear`: Linear model
- `mlp`: Multi-layer perceptron with 2 hidden layer (128 units, 64 units)

