# train.sh

#!/bin/bash

#Train LeNet on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model LeNet --dataset MNIST --opt SGD

#Train LeNet on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model LeNet --dataset MNIST --opt Adam

# Train ResNet-18 on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model ResNet --dataset CIFAR10 --opt SGD

# Train ResNet-18 on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model ResNet --dataset CIFAR10 --opt Adam




