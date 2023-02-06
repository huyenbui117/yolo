"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import (
    mean_average_precision,
    get_bboxes
)
from loss import YoloLoss
import os

seed = 0
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "mps" if getattr(torch, "mps", False) else "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8# 64 in original paper
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 0
PIN_MEMORY = True
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
PRETRAIN = "./yolov1.pt"

transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)


def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item(), epoch=epoch)

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_dataset = VOCDataset(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    if not os.path.exists(PRETRAIN) or PRETRAIN == None:
        for epoch in range(EPOCHS):
            
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.1, threshold=0.1, device=DEVICE
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.1, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec}")

            train_fn(train_loader, model, optimizer, loss_fn, epoch)
            
        # save model
        torch.save(model.state_dict(), "yolov1.pt")
        print(f'Checkpoint saved at "yolov1.pt"')
    else:
        model.load_state_dict(torch.load(PRETRAIN))
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.1, threshold=0.1, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.1, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

    pred_boxes, target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.1, threshold=0.1, device=DEVICE
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.1, box_format="midpoint"
    )
    print(f"Test mAP: {mean_avg_prec}")


if __name__ == "__main__":
    main()