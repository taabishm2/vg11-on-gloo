import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl

from measure import *

device = "cpu"
torch.set_num_threads(4)
torch.manual_seed(744)

batch_size = 256 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    running_loss = total_loss = 0.0

    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Batch:", batch_idx, end=" ")
        save_params("main.params", model)
        
        t1 = time.time()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data, target

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        measure_iters(source="baseline", iter=batch_idx, start_time=t1, 
                      iter_loss=loss.item(), total_loss=total_loss/(batch_idx+1), 
                      batch_size=inputs.size(0), sync_time=0)

    print('Finished Training')

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # Pad 4 px all side, random 32x32 crop for diversity
            transforms.RandomHorizontalFlip(), # Randomly flip image for diversity
            transforms.ToTensor(),
            normalize,
            ])

    # No random crop and horizontal flip for testing
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    # num_workers refers to the number of workers to use for data loading
    # sampler denotes how to sample from the dataset. If None, we use random sampling
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    shuffle=True,
                                                    pin_memory=True)
    
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    # running training for one epoch
    for epoch in range(1):
        tt1 = time.time()
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        print("**** TOTAL TRAIN TIME: ", time.time()-tt1)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
