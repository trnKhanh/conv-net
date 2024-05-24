import torch


def save_checkpoint(save_path, epoch, model, optimizer, lr_scheduler):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
    }
    torch.save(ckpt, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = 0
    if "epoch" in ckpt:
        epoch = ckpt["epoch"]
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["scheduler"])

    print(f"Loaded checkpoint from {ckpt_path}")

    return epoch + 1
