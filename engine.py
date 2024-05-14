from tqdm import tqdm

import torch
import numpy as np


def train_one_epoch(epoch, model, optimizer, loss_fn, dataloader, device):
    model.train()
    correct_count = 0
    total_count = 0
    loss_values = []
    with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for samples, labels in tepoch:
            # Move tensor to device used to train
            samples = samples.to(device)
            labels = labels.to(device)
            
            # Training loop in pytorch
            preds = model(samples)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute average loss and accuracy
            loss_value = loss.item()
            pred_classes = torch.argmax(preds, dim=1)
            correct = torch.sum(pred_classes == labels).item()

            correct_count += correct
            total_count += len(samples)
            acc = correct_count / total_count

            loss_values.append(loss_value)

            tepoch.set_postfix(
                dict(acc=acc, loss=torch.mean(torch.Tensor(loss_values)).item())
            )
    return torch.mean(torch.Tensor(loss_values)).item()


def valid_one_epoch(model, loss_fn, train_dataloader, valid_dataloader, device):
    model.eval()
    loss_values = []
    correct_count = 0
    total_count = 0
    acc = 0
    _pred_classes = dict() 
    _labels = dict()
    datasets = []
    if train_dataloader is not None:
        datasets.append(("Train", train_dataloader))
    if valid_dataloader is not None:
        datasets.append(("Test", valid_dataloader))
    for name, dataloader in datasets:
        _pred_classes[name] = []
        _labels[name] = []
        with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
            tepoch.set_description(f"{name} validation")
            with torch.no_grad():
                for samples, labels in tepoch:
                    # Move tensor to device used to train
                    samples = samples.to(device)
                    labels = labels.to(device)
                    preds = model(samples)
                    loss = loss_fn(preds, labels)

                    # Compute average loss and accuracy
                    loss_value = loss.item()
                    pred_classes = torch.argmax(preds, dim=1)
                    _pred_classes[name].extend(pred_classes.tolist())
                    _labels[name].extend(labels.tolist())
                    correct = torch.sum(pred_classes == labels).item()
                    correct_count += correct
                    total_count += len(samples)
                    acc = correct_count / total_count

                    loss_values.append(loss_value)

                    tepoch.set_postfix(
                        dict(
                            acc=acc,
                            loss=torch.mean(torch.Tensor(loss_values)).item(),
                        )
                    )
    
    return acc, torch.mean(torch.Tensor(loss_values)).item(), _pred_classes, _labels
