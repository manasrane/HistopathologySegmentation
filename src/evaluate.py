import torch
import numpy as np
from sklearn.metrics import jaccard_score

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    return (pred == target).mean()

def evaluate_model(model, loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    accuracies = []

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred = (pred > 0.5).float()

            dice = dice_coefficient(pred.cpu().numpy(), mask.cpu().numpy())
            iou = iou_score(pred.cpu().numpy(), mask.cpu().numpy())
            acc = pixel_accuracy(pred.cpu().numpy(), mask.cpu().numpy())

            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracies.append(acc)

    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'accuracy': np.mean(accuracies)
    }