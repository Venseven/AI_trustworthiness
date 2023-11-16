import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models, transforms


def craft_random_lflip(train_set, ratio):
    num_poisons = int(len(train_set) * ratio)
    indices_to_poison = np.random.choice(len(train_set), num_poisons, replace=False)
    print(indices_to_poison)

    mask = torch.zeros_like(train_set.targets, dtype=torch.bool)
    mask[indices_to_poison] = True
    train_set.targets[mask] = 1 - train_set.targets[mask]

    return train_set


def craft_clabel_poisons(model, target, bases, niter, lr, beta, target_num):
    # Ensure the model is in evaluation mode
    model.eval()
    # Create a copy of the bases as the initial poisons
    poisons = bases.clone().detach().requires_grad_(True)

    # Use an optimizer such as SGD
    criterion = nn.MSELoss()
    optimizer = optim.SGD([poisons], lr=lr)
    target_features = model(target).detach()


    target_100_10 = target_features.repeat(100, 1)

    for i in range(niter):
        optimizer.zero_grad()

        # Calculate the feature representations of the poisons
        poison_features = model(poisons)

        # Calculate the loss
        # loss = ((poison_features - target_features) ** 2).sum() + beta * ((poisons - bases) ** 2).sum()
        target_loss = criterion(poison_features, target_100_10)
        bases_loss = criterion(poisons, bases)
        loss = (beta*target_loss) + (bases_loss)
        # print(target_loss, bases_loss, loss)

        # Backpropagate the gradients
        loss.backward(retain_graph=True)

        # Update the poisons
        optimizer.step()
        if i % 50 == 0:
            print(f'Iteration: {i} , Loss: {loss}, Base Loss : {bases_loss}, target_loss : {target_loss}')

    poisons = torch.clamp(poisons, min=0, max=1)

    image_pil = transforms.ToPILImage()(poisons[0])
    image_name = "poison_" + str(target_num) +".png"
    image_pil.save(image_name)

    image_pil = transforms.ToPILImage()(bases[0])
    image_name = "base_" + str(target_num) + ".png"
    image_pil.save(image_name)

    image_pil = transforms.ToPILImage()(target.squeeze())
    image_name = "target_" + str(target_num) + ".png"
    image_pil.save(image_name)

    return poisons.detach()
