import torch
from torchvision import datasets, transforms

def load_mnist(batch_size=32, rotation=0, hflip=False):
    transform_list = [transforms.ToTensor()]
    
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation))
    
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    transform = transforms.Compose(transform_list)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def load_cifar10(batch_size=32, rotation=0, hflip=False):
    transform_list = [transforms.ToTensor()]
    
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation))
    
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    transform = transforms.Compose(transform_list)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
