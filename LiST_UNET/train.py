




import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import DataLoader

from LiST_UNET import LiST_UNET
from loss import Sum as LossBuilder
from mydataset_3DMRF import MRFdataset
from optimizer import LinearWarmupCosineAnnealingLR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiST-UNet for 3D MRF reconstruction")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--loss_name", type=str, default="L1retest", help="Loss function name")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="",
        help="Dataset root directory",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--max_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Inference result output directory",
    )
    parser.add_argument("--person", type=int, default=0, help="Person index")
    return parser.parse_args()


def build_model(device: torch.device) -> torch.nn.Module:
    model = LiST_UNET(
        in_channels=5,
        out_channels=1,
        embed_dim=96,
        embedding_dim=1152,
        channels=(32, 64, 128),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    )
    return model.to(device)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, list]:
    train_dataset = MRFdataset()
    test_dataset = MRFdataset()

    train_dataset.getpath(args, Dp=1, person=args.person)
    save_path = test_dataset.getpath(args, Dp=0, person=args.person)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )
    return train_loader, test_loader, save_path


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> float:
    model.train()
    epoch_losses = []

    for step, batch in enumerate(train_loader):
        inputs, labels, masks, _ = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(inputs)
        loss_builder = LossBuilder(loss_name, preds, labels, masks)
        loss_tensor = loss_builder.add()

        loss_tensor.backward()
        optimizer.step()

        loss_value = loss_tensor.item()
        epoch_losses.append(loss_value)

        print(
            f"[Train] Epoch [{epoch + 1}/{max_epochs}] "
            f"Step [{step + 1}/{len(train_loader)}] "
            f"Loss: {loss_value:.4f}"
        )

    return float(np.mean(epoch_losses))


def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_name: str,
    device: torch.device,
    result_root: Path,
    epoch: int,
) -> None:
    model.eval()
    result_root.mkdir(parents=True, exist_ok=True)

    loss_list = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs, labels, masks, paths = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            preds = model(inputs)

            loss_builder = LossBuilder(loss_name, preds, labels, masks)
            loss_tensor = loss_builder.add()
            loss_value = loss_tensor.item()
            loss_list.append(loss_value)

            print(f"[Test] Sample [{idx + 1}/{len(test_loader)}] Loss: {loss_value:.4f}")
            print(paths)

            save_img = torch.squeeze(preds).cpu().numpy()
            save_label = torch.squeeze(labels).cpu().numpy()
            save_mask = torch.squeeze(masks).cpu().numpy()

            io.savemat(result_root / "loss.mat", {"loss": loss_list})
            io.savemat(
                result_root / "result.mat",
                {
                    "input": save_img,
                    "mask": save_mask,
                    "label": save_label,
                },
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_dir = Path(args.ckpt_dir)
    result_dir = Path(args.result_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, save_path = build_dataloaders(args)

    model = build_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.max_epochs,
    )

    for epoch in range(args.max_epochs):
        mean_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_name=args.loss_name,
            device=device,
            epoch=epoch,
            max_epochs=args.max_epochs,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch Summary] epoch={epoch + 1}, lr={current_lr:.8f}, mean_loss={mean_loss:.6f}")

        ckpt_path = ckpt_dir / f"modelpara_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)

        if epoch > 0:
            epoch_result_dir = result_dir / save_path[0] / f"{epoch}"
            run_inference(
                model=model,
                test_loader=test_loader,
                loss_name=args.loss_name,
                device=device,
                result_root=epoch_result_dir,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()