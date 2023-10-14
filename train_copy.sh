# train.sh

#!/bin/bash

#Train LeNet on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model LeNet --dataset MNIST --lr 0.001

#Train LeNet on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model LeNet --dataset MNIST --lr 0.0001

# Train ResNet-18 on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model ResNet --dataset CIFAR10 --lr 0.001

# Train ResNet-18 on MNIST dataset
python3 /scratch/subramav/AI_trustworthiness/train1.py --model ResNet --dataset CIFAR10 --lr 0.0001




