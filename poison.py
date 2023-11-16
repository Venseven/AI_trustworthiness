# Import statements
import os
import copy
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression

# Import custom modules
from datasets import load_mnist, load_cifar10

from models.logistic_regression import LogisticRegression
# from models.resnet18 import ResNet18
from poison_craft import craft_random_lflip, craft_clabel_poisons

# # Set environment variables
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_configuration():
    # Define and return configuration parameters
    config = {
        'rotation': 0,
        'hflip': 0,
        'optimizer_name': 'sgd',
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'save_name': '',
        'task': 1
    }
    return config

def train_and_test_classification_model():
    config = load_configuration()

    if config['task'] == 1:
        train_and_test_mnist_classification_model(config)
    else:
        train_and_test_cifar_classification_model(config)

def train_and_test_mnist_classification_model(config):
    # Load MNIST-1/7 dataset
    train_val_dataset, test_dataset = load_mnist(config['batch_size'], config['rotation'], config['hflip'])
    ratios = [0, 0.05, 0.1, 0.25, 0.5]

    input_dim = 28 * 28
    output_dim = 2

    model = LogisticRegression(input_dim)
    model_name = 'LogReg'
    dataset_name = 'mnist'

    classification_accuracy_plot = []

    for ratio in ratios:
        config['save_name'] = str(ratio)
        poisoned_dataset = craft_random_lflip(train_val_dataset.dataset, ratio)

        # Training Validation Split
        train_size = int(0.9 * len(poisoned_dataset))
        val_size = len(poisoned_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset=poisoned_dataset,
                                                                   lengths=[train_size, val_size])

        # Creating Dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True)

        trained_model = train_classification_model(model_name, dataset_name, model, train_dataloader,
                                                   val_dataloader, config)

        classification_accuracy = test_classification_model(model_name, dataset_name, trained_model, test_dataloader)
        classification_accuracy_plot.append(classification_accuracy)

    # Plotting and saving the plots
    plot_dir_s = Path("plots")
    plot_dir_s.mkdir(parents=True, exist_ok=True)
    plot_name = "task1.png"
    plot_save_path = plot_dir_s / plot_name

    plt.plot(ratios, classification_accuracy_plot, marker='o')
    plt.title('Classification Accuracy vs Ratio of Poisoned Samples')
    plt.xlabel('Ratio of Poisoned Samples')
    plt.ylabel('Classification Accuracy')
    plt.savefig(plot_save_path)

def train_classification_model(model_name, dataset_name, model, train_dataloader, val_dataloader, config):
    # Training Parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['learning_rate'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = model.to(device)

    # Experiment tracking
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = dataset_name
    model_name = model_name
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    writer = SummaryWriter(log_dir)

    EPOCHS = 30

    train_acc_plt = []
    val_acc_plt = []
    train_loss_plt = []
    val_loss_plt = []

    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            # model.train()
            # Forward pass
            output = model(data)
            loss = loss_fn(output.squeeze(), target.float())

            # y_pred = model(X)
            # y = y.view(-1, 1)
            # y_pred = y_pred.view(-1, 1)
            # y = y.to(dtype=torch.float)
            # y_pred = y_pred.to(dtype=torch.int64)

            # loss = loss_fn(y_pred, y)

            train_loss += loss.item()

            # acc = accuracy(y_pred, y)
            # train_acc += acc

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = (torch.sigmoid(output) > 0.5).float()

            for i, prediction in enumerate(predictions):
                if prediction == target[i]:
                    correct += 1
                total += 1

        train_loss /= total
        train_acc = correct / total

        if epoch % 2 == 0:

            # Validation loop
            val_loss, val_acc = 0.0, 0.0
            model.eval()
            with torch.inference_mode():
                val_corr = 0
                val_tot = 0
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)

                    y_pred = model(X)

                    loss = loss_fn(y_pred.squeeze(), y.float())
                    val_loss += loss.item()

                    # Calculate accuracy
                    predictions = (torch.sigmoid(y_pred) > 0.5).float()

                    for i, prediction in enumerate(predictions):
                        if prediction == y[i]:
                            val_corr += 1
                        val_tot += 1

                val_loss /= val_tot
                val_acc = val_corr / val_tot

            train_acc_plt.append(train_acc)
            val_acc_plt.append(val_acc)
            train_loss_plt.append(train_loss)
            val_loss_plt.append(val_loss)

            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc},
                               global_step=epoch)

            print(
                f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")

    model_dir_s = Path("models")
    model_dir_s.mkdir(parents=True, exist_ok=True)

    model_name_s = model_name + "_" + dataset_name + "_" + config['save_name']
    model_save_path = model_dir_s / model_name_s

    # Saving the model
    print(f"Saving the model: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    return model

def test_classification_model(model_name, dataset_name, model, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Test_set")
    model = model.to(device)

    accuracy = Accuracy(task='multiclass', num_classes=2)
    accuracy = accuracy.to(device)

    # Experiment tracking
    timestamp = datetime.now().strftime("%Y-%m-%d")
    experiment_name = dataset_name
    model_name = model_name
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    writer = SummaryWriter(log_dir)

    EPOCHS = 5

    test_acc_avg = 0.0

    for epoch in range(EPOCHS):
        # Testing loop
        test_acc = 0.0
        model.eval()
        with torch.inference_mode():
            correct = 0
            total = 0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)

                y_pred = model(X)
                # Calculate accuracy
                predictions = (torch.sigmoid(y_pred) > 0.5).float()

                for i, prediction in enumerate(predictions):
                    if prediction == y[i]:
                        correct += 1
                    total += 1

            test_acc = correct / total

        test_acc_avg = ((test_acc_avg * epoch) + test_acc) / (epoch + 1)

        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"test/acc": test_acc},
                           global_step=epoch)

        print(f"Epoch: {epoch}| test acc: {test_acc: .5f} | test acc avg: {test_acc_avg: .5f}")
    return test_acc_avg

def contaminated_finetuning(model, contaminated_trainloader, epochs, target_image):
    model_copy = copy.deepcopy(model)
    # Define the loss function and the optimizer for the last layer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.final_layers.parameters(), lr=0.01)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_copy = model_copy.to(device)
    for epoch in range(epochs):
        # Set the model to training mode
        model_copy.train()
        # Initialize the running loss
        running_loss = 0.0
        # Loop over the batches of the contaminated training set
        correct = 0
        total = 0
        train_loss, train_acc = 0.0, 0.0
        for batch_idx, (data, target) in enumerate(contaminated_trainloader):
            data = data.to(device)
            target = target.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model_copy(data)

            # Compute the loss
            loss = criterion(outputs, target)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Update the running loss
            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            for i, prediction in enumerate(predictions):
                if prediction == target[i]:
                    correct += 1
                total += 1
        train_acc = correct / total
        train_loss = running_loss / total
        print(f"[{epoch + 1}] loss: {train_loss} acc: {train_acc}")

    model_copy.to('cpu')
    model_copy.eval()
    print(model_copy(target_image))
    _, predicted_class = torch.max(model_copy(target_image), 1)

    return predicted_class

if __name__ == "__main__":
    train_and_test_classification_model()
