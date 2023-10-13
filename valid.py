'a Python script for evaluating a pre-trained model'

import torch
from datasets import load_mnist, load_cifar10
import matplotlib.pyplot as plt
from models.lenet import LeNet
from models.resnet import ResNet18
from models.vgg import VGG16

# Choose your model and dataset here
model = LeNet()  # Change this to VGG16() or ResNet18() as needed
train_loader, test_loader = load_mnist()  # Change this to load_cifar10() for CIFAR-10

# Load pre-trained weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

#  train 3 DNNs (LeNet, VGG16, and ResNet18) on 2 datasets (MNIST and CIFAR-10). You need to measure your model's performance with two metrics: classification accuracy and loss. You can compute them on both the training and testing data.

# Please compute those metrics in every 5 training iterations (epochs). Draw 2 plots for each model training: { epochs } vs. { training accuracy & testing accuracy } and { epochs } vs. { training loss & testing loss }

