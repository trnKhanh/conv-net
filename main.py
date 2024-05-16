from argparse import ArgumentParser
import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from net import Net
from dataset import get_dataset
from engine import train_one_epoch, valid_one_epoch

from PIL import Image


def create_args():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--eval-log-dir", default="", type=str)
    # Dataset
    parser.add_argument("--hog", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "FashionMNIST", "Caltech101", "Caltech256"],
    )

    # Training
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--fig-dir", default="", type=str)
    parser.add_argument("--patience", default=10, type=int)

    # Checkpoints
    parser.add_argument("--save-path", default="", type=str)
    parser.add_argument("--save-freq", default=5, type=int)
    parser.add_argument("--save-best-path", default="", type=str)

    parser.add_argument("--load-ckpt", default="", type=str)

    return parser.parse_args()


def main(args):
    # Build dataset based on arguments specified by user
    num_classes, train_dataset, valid_dataset = get_dataset(args.dataset)

    print("=" * os.get_terminal_size().columns)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(valid_dataset)} samples")
    print("=" * os.get_terminal_size().columns)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(args.device)

    # Build neural network using arguments passed by user
    model = Net(3, num_classes)
    print("=" * os.get_terminal_size().columns)
    print(model)
    params = sum([p.numel() for p in model.parameters()])
    print(f"Parameters: {params}")
    print("=" * os.get_terminal_size().columns)
    model.to(device)
    if len(args.load_ckpt):
        state_dict = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.load_ckpt}")

    loss_fn = nn.CrossEntropyLoss()
    if args.train:
        train_loss_values = []
        valid_loss_values = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if len(args.save_path) > 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if len(args.save_best_path) > 0:
            os.makedirs(os.path.dirname(args.save_best_path), exist_ok=True)
        if len(args.fig_dir) > 0:
            os.makedirs(args.fig_dir, exist_ok=True)
        min_loss = math.inf
        not_better = 0
        # Loop and train for n epochs
        for e in range(1, args.epochs + 1):
            # Train and update model's parameters
            train_loss = train_one_epoch(
                e, model, optimizer, loss_fn, train_dataloader, device
            )
            # Validating model using test set
            acc, valid_loss, _, _ = valid_one_epoch(
                model, loss_fn, None, valid_dataloader, device
            )
            train_loss_values.append(train_loss)
            valid_loss_values.append(valid_loss)

            # Save best checkpoint based on loss value
            if len(args.save_best_path) > 0 and valid_loss < min_loss:
                torch.save(model.state_dict(), args.save_best_path)
                print(f"Saved model to {args.save_best_path}")
            if min_loss <= valid_loss:
                not_better += 1
            else:
                not_better = 0

            min_loss = min(min_loss, valid_loss)
            if not_better == args.patience:
                print(f"Stopping because of exceeding patience")
                break

            # Save every freq (default: 5) epochs
            if len(args.save_path) > 0 and (
                e % args.save_freq == 0 or e == args.epochs
            ):
                torch.save(model.state_dict(), args.save_path)
                print(f"Saved model to {args.save_path}")
        # The following lines of code used for visualizing loss during training
        if len(args.fig_dir) > 0:
            x = np.arange(1, len(train_loss_values) + 1, dtype=np.int32)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(
                x,
                np.array(train_loss_values),
                color="g",
                label="train",
            )
            plt.plot(
                x,
                np.array(valid_loss_values),
                color="r",
                label="valid",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                args.fig_dir
                + "/"
                + args.save_path.split("/")[-1].split(".")[-2]
                + ".pdf"
            )
    else:
        acc, loss, preds, labels = valid_one_epoch(
            model, loss_fn, train_dataloader, valid_dataloader, device
        )
        if len(args.eval_log_dir) > 0:
            os.makedirs(os.path.dirname(args.eval_log_dir), exist_ok=True)
            with open(args.eval_log_dir, "w") as f:
                json.dump({"preds": preds, "labels": labels}, f)


if __name__ == "__main__":
    args = create_args()
    main(args)
