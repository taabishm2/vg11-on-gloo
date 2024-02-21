import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
device = "cpu"
torch.set_num_threads(4)
from measure import *

batch_size = 256 # batch for one node
world_size=1
rank=0

# Set the random seed
torch.manual_seed(744)

def all_reduce_params(model):
    # Initialize a list to store gradients from all parameters
    all_grads = []
    
    # Iterate over all parameters in the model
    for param in model.parameters():
        # Ensure gradients are not None
        if param.grad is not None:
            # Append the gradients to the list
            all_grads.append(param.grad.data.view(-1))

    output_tensor = torch.zeros(len(all_grads))

   # Create a single tensor containing all gradients from all workers
    all_grads_concatenated = torch.cat(all_grads)
    # all_grads_concatenated = torch.stack(all_grads, dim=0)

    # Initialize a tensor to store the mean of gradients across all workers
    mean_grads = torch.zeros_like(all_grads_concatenated)

    # print(len(all_grads), "all_grads_concatenated ", all_grads_concatenated.size(), " mean_grads ", mean_grads.size())

    # Rank 0 gathers all gradients from all workers
    if rank == 0:
        gather_list = [torch.zeros_like(all_grads_concatenated) for _ in range(dist.get_world_size())]
    else:
        gather_list = None

    dist.gather(all_grads_concatenated, gather_list=gather_list, dst=0)

    # Rank 0 calculates the mean of gradients
    if rank == 0:
        mean_grads = torch.mean(torch.stack(gather_list), dim=0)

    # Rank 0 scatters the mean gradients to all workers
    if rank == 0:
        scatter_list = [mean_grads.clone() for _ in range(dist.get_world_size())]
    else:
        scatter_list = None
    dist.scatter(mean_grads, scatter_list, src=0)

    # Each worker updates its own gradients with the received mean gradients
    for param, mean_grad in zip(model.parameters(), mean_grads):
        if param.grad is not None:
            param.grad.data.copy_(mean_grad)

    # Synchronize to ensure all workers have updated gradients before continuing
    dist.barrier()

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        print("Batch:", batch_idx, end=" ")
        save_params("2b_first.params", model)
        
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        # Perform gradient aggregation
        all_reduce_params(model)

        optimizer.step()

        if batch_idx % 20 == 0:
            print('Iteration: {} \tLoss: {:.6f}'.format(
                batch_idx, loss.item()))

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

# code for 2a

def setup_distributed(args):
    # Initialize the process group
    # dist.init_process_group(backend='gloo', init_method="tcp://172.18.0.2:12345,172.18.0.3:12345,172.18.0.4:12345,172.18.0.5:12345", rank=0, world_size=4)
    dist.init_process_group(backend='gloo', init_method=args.master_ip,
                            world_size=args.num_nodes, rank=args.rank)
    global rank, world_size
    rank = args.rank
    world_size=args.num_nodes

def cleanup_distributed():
    # Clean up the process group
    dist.destroy_process_group()

def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--master-ip', type=str, default='172.18.0.2',
                        help='IP of the master node')
    parser.add_argument('--num-nodes', type=int, default=4,
                        help='Number of nodes in the cluster')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of the current node')
    args = parser.parse_args()

    setup_distributed(args)
    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)

    sampler = DistributedSampler(dataset=training_set, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
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
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    main()
