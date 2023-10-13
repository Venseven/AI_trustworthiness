import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_mnist, load_cifar10
from models.lenet import LeNet
from models.resnet import ResNet18
from models.vgg16 import VGG16
import ray
from ray import air

from ray import tune
ray.init(_temp_dir = "/scratch/subramav/tmps")
def train_model(config):
    # Choose your model and dataset based on hyperparameters
    if config['model'] == 'LeNet':
        model = LeNet(dataset=config['dataset'])
    elif config['model'] == 'VGG16':
        model = VGG16(dataset=config['dataset'])
    elif config['model'] == 'ResNet':
        model = ResNet18(dataset=config['dataset'])
    
    if config['dataset'] == 'MNIST':
        train_loader, test_loader = load_mnist()
    elif config['dataset'] == 'CIFAR10':
        train_loader, test_loader = load_cifar10()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define your optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    criterion = F.cross_entropy

    # Training loop
    epochs = config['epochs']
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Implement the validation here, and report the metrics you want to tune
        model.eval()
        validation_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                validation_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        
        # Report metrics
        tune.report(loss=validation_loss, accuracy=accuracy)

# Define the configuration/hyperparameter space
config = {
    "model": tune.grid_search(['LeNet', 'VGG16', 'ResNet']),
    "dataset": tune.grid_search(['MNIST', 'CIFAR10']),
    "lr": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.1, 0.9),
    "epochs": tune.choice([5, 10, 15])
}

analysis = tune.run(
    train_model,
    config=config,
    resources_per_trial={"cpu":0,"gpu": 1},  # Adjust the resources here based on your machine
    num_samples=2,
    local_dir = "/scratch/subramav/results"# Number of trials
)

print("Best trial config: ", analysis.get_best_config(metric="accuracy", mode="max"))
