from argparse import ArgumentParser
import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from net import Net
from dataset import get_dataset
from optim import CosineSchedule
from engine import train_one_epoch, valid_one_epoch
from utils import load_checkpoint, save_checkpoint

from PIL import Image
import yaml


def create_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="Whether to train model"
    )
    parser.add_argument(
        "--valid", action="store_true", help="Whether to run validation"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to run model on"
    )
    parser.add_argument(
        "--eval-log-dir",
        default="",
        type=str,
        help="Path to evaluation result file",
    )
    # Model
    parser.add_argument(
        "--transfer-learning-num-classes",
        default=0,
        type=int,
        help="Number of classes of pretained model",
    )
    parser.add_argument(
        "--transfer-learning",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to model config file",
    )
    parser.add_argument(
        "--mlp-dropout-rate",
        default=0,
        type=int,
        help="Dropout rate of linear layers",
    )
    parser.add_argument(
        "--conv-dropout-rate",
        default=0,
        type=int,
        help="Dropout rate of convolution layers",
    )
    # Dataset
    parser.add_argument(
        "--split-path",
        default="",
        type=str,
        help="Path to file containing information about how to split dataset. If not specify, then dataset is random splitted",
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "FashionMNIST", "Caltech101", "Caltech256"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of workers in dataloader",
    )

    # Training
    parser.add_argument(
        "--epochs", default=100, type=int, help="How many epochs to train model"
    )
    parser.add_argument(
        "--start-epochs", default=1, type=int, help="Epoch to start from"
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Batch size used to feed into models",
    )
    parser.add_argument(
        "--base-lr",
        default=0.005,
        type=float,
        help="Base/Initial learning rate",
    )
    parser.add_argument(
        "--target-lr",
        default=0.00001,
        type=float,
        help="Target/Final learning rate",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=5,
        type=int,
        help="How many epochs are used for warming up",
    )
    parser.add_argument(
        "--max-epochs",
        default=70,
        type=int,
        help="How many epochs to decay learning rate",
    )
    parser.add_argument(
        "--fig-dir",
        default="",
        type=str,
        help="Directory to save figures/plots about training process",
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="Number of epochs without improvement to wait before stopping train",
    )

    # Checkpoints
    parser.add_argument(
        "--save-path", default="", type=str, help="Where to save checkpoint"
    )
    parser.add_argument(
        "--save-freq",
        default=5,
        type=int,
        help="How frequent to save checkpoint",
    )
    parser.add_argument(
        "--save-best-path",
        default="",
        type=str,
        help="Where to save best checkpoint (according to loss)",
    )
    parser.add_argument(
        "--save-best-acc-path",
        default="",
        type=str,
        help="Where to save best checkpoint (according to accuracy)",
    )

    parser.add_argument(
        "--load-ckpt", default="", type=str, help="Where to load checkpoint"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="Resume training from specified checkpoint",
    )

    return parser.parse_args()


def main(args):
    # Build dataset based on arguments specified by user
    num_classes, train_dataset, valid_dataset = get_dataset(
        args.dataset, args.split_path
    )

    print("=" * os.get_terminal_size().columns)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(valid_dataset)} samples")
    print(f"Total: {len(valid_dataset) + len(train_dataset)} samples")
    print("=" * os.get_terminal_size().columns)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Declare how to transform images in train/valid set
    image_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(224),
            ]
        ),
    }

    device = torch.device(args.device)

    # Build neural network using arguments passed by user
    with open(args.model, "r") as f:
        model_config = yaml.safe_load(f)
        net_configs = [_ for _ in range(len(model_config["model"]["net"]))]
        for k, v in model_config["model"]["net"].items():
            net_configs[k] = v

        if "init_down" in model_config["model"]:
            init_down = [
                _ for _ in range(len(model_config["model"]["init_down"]))
            ]
            for k, v in model_config["model"]["init_down"].items():
                init_down[k] = v
        else:
            init_down = None

        mlp_configs = model_config["model"]["mlp"]

    model = Net(
        3,
        (
            num_classes
            if args.transfer_learning_num_classes == 0
            else args.transfer_learning_num_classes
        ),
        net_configs,
        mlp_configs,
        args.mlp_dropout_rate,
        args.conv_dropout_rate,
        model_config["model"]["max_pool_stride"],
        init_down=init_down,
    )
    model.to(device)
    # Load checkpoint if user specifies
    if len(args.load_ckpt):
        state_dict = torch.load(args.load_ckpt, map_location=device)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model from {args.load_ckpt}")

    loss_fn = nn.CrossEntropyLoss()
    if args.train:
        if len(args.transfer_learning):
            state_dict = torch.load(args.transfer_learning, map_location=device)
            if "model" in state_dict:
                model.load_state_dict(state_dict["model"])
            else:
                model.load_state_dict(state_dict)
            for param in model.parameters():
                param.requires_grad = False
            model.create_head(num_classes, mlp_configs)
            print(f"transfer learning from {args.transfer_learning}")
        print("=" * os.get_terminal_size().columns)
        summary(model, (3, 224, 224), args.batch_size, args.device)
        print("=" * os.get_terminal_size().columns)

        train_loss_values = []
        valid_loss_values = []
        # Declare optimizer and lr scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
        )

        scheduler = CosineSchedule(
            optimizer,
            base_lr=args.base_lr,
            target_lr=args.target_lr,
            max_steps=args.max_epochs,
            warmup_steps=args.warmup_epochs,
        )
        # Resume training if user specifies
        if len(args.resume):
            args.start_epochs = load_checkpoint(
                args.resume, model, optimizer, scheduler, args.device
            )

        if len(args.save_path) > 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if len(args.save_best_path) > 0:
            os.makedirs(os.path.dirname(args.save_best_path), exist_ok=True)
        if len(args.fig_dir) > 0:
            os.makedirs(args.fig_dir, exist_ok=True)

        min_loss = math.inf
        max_acc = -math.inf
        not_better = 0
        # Loop and train for n epochs
        for e in range(args.start_epochs, args.epochs + 1):
            # Train and update model's parameters
            train_loss = train_one_epoch(
                e,
                model,
                optimizer,
                loss_fn,
                train_dataloader,
                device,
                transform=image_transforms["train"],
                lr_scheduler=scheduler,
            )
            # Validating model using test set
            acc, valid_loss, _, _ = valid_one_epoch(
                model,
                loss_fn,
                None,
                valid_dataloader,
                device,
                transform=image_transforms["valid"],
            )
            train_loss_values.append(train_loss)
            valid_loss_values.append(valid_loss["Test"])

            # Save best checkpoint based
            if len(args.save_best_path) > 0 and valid_loss["Test"] < min_loss:
                save_checkpoint(
                    args.save_best_path,
                    e,
                    model,
                    optimizer,
                    scheduler,
                )
            if len(args.save_best_acc_path) > 0 and acc["Test"] > max_acc:
                save_checkpoint(
                    args.save_best_acc_path,
                    e,
                    model,
                    optimizer,
                    scheduler,
                )
            if min_loss <= valid_loss["Test"] and max_acc >= acc["Test"]:
                not_better += 1
            else:
                not_better = 0

            min_loss = min(min_loss, valid_loss["Test"])
            max_acc = max(max_acc, acc["Test"])

            # If exceeding patience, then stop training and save checkpoint
            if not_better == args.patience:
                print(f"Stopping because of exceeding patience")
                if len(args.save_path) > 0:
                    save_checkpoint(
                        args.save_path,
                        e,
                        model,
                        optimizer,
                        scheduler,
                    )
                break

            # Save checkpoint
            if len(args.save_path) > 0 and (
                e % args.save_freq == 0 or e == args.epochs
            ):
                save_checkpoint(
                    args.save_path,
                    e,
                    model,
                    optimizer,
                    scheduler,
                )
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
    # Run evaluation on both train and valid set
    if args.valid:
        acc, loss, preds, labels = valid_one_epoch(
            model,
            loss_fn,
            train_dataloader,
            valid_dataloader,
            device,
            transform=image_transforms["valid"],
        )
        if len(args.eval_log_dir) > 0:
            os.makedirs(os.path.dirname(args.eval_log_dir), exist_ok=True)
            with open(args.eval_log_dir, "w") as f:
                json.dump({"preds": preds, "labels": labels}, f)


if __name__ == "__main__":
    args = create_args()
    main(args)
